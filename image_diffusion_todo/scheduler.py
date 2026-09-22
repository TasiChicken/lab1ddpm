from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.register_buffer(
            "timesteps",
            torch.from_numpy(
                np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
            ),
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        elif mode == "cosine":
            ######## TODO ########
            # Implement the cosine beta schedule (Nichol & Dhariwal, 2021).
            # Hint:
            # 1. Define alphā_t = f(t/T) where f is a cosine schedule:
            #       alphā_t = cos^2( ( (t/T + s) / (1+s) ) * (π/2) )
            #    with s = 0.008 (a small constant for stability).
            # 2. Convert alphā_t into betas using:
            #       beta_t = 1 - alphā_t / alphā_{t-1}
            # 3. Clip beta_t to at most 0.999 (singularity at t = T).
            # 4. Return betas as a tensor of shape [num_train_timesteps].
            s = 0.008

            steps = torch.arange(
                num_train_timesteps + 1,
                dtype=torch.float32
            )

            f = torch.cos(
                ((steps / num_train_timesteps + s) / (1 + s))
                * torch.pi / 2
            ) ** 2

            alpha_bar = f / f[0]

            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]

            betas = torch.clamp(
                betas,
                min=0.0,
                max=0.999
            )
               
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
        
        self.schedule_mode = mode      

        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    
    
    def step(self, x_t: torch.Tensor, t: int, net_out: torch.Tensor, predictor: str):
        # Normalise t once here so each step_predict_* below receives a 1-D
        # tensor on x_t's device, whether it was given a python int or the
        # 0-dim tensor sample() iterates over.
        if isinstance(t, int):
            t = torch.tensor([t])
        t = t.reshape(-1).to(x_t.device)

        if predictor == "noise": #### TODO
            return self.step_predict_noise(x_t, t, net_out)
        elif predictor == "x0": #### TODO
            return self.step_predict_x0(x_t, t, net_out)
        elif predictor == "mean": #### TODO
            return self.step_predict_mean(x_t, t, net_out)
        else:
            raise ValueError(f"Unknown predictor: {predictor}")

    
    def step_predict_noise(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        Noise prediction version (the standard DDPM formulation).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            eps_theta: predicted noise ε̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 1. Extract beta_t, alpha_t, alpha_bar_t, and alpha_bar_{t-1} from the
        #    scheduler (ᾱ_{t-1} = 1 at t = 0).
        # 2. Convert the predicted noise into the predicted clean sample
        #       x̂₀ = (x_t - √(1-ᾱ_t) * ε̂_θ) / √ᾱ_t
        #    and clamp it to [-1, 1].
        # 3. Compute the posterior mean
        #       \tilde{μ}_t = (√ᾱ_{t-1}·β_t/(1-ᾱ_t)) * x̂₀ + (√α_t·(1-ᾱ_{t-1})/(1-ᾱ_t)) * x_t.
        # 4. Compute the posterior variance \tilde{β}_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t.
        # 5. Add Gaussian noise scaled by √(\tilde{β}_t) unless t == 0.
        # 6. Return the final sample at t-1.
        beta_t = extract(
            self.betas, t, x_t
        )

        alpha_t = extract(
            self.alphas, t, x_t
        )

        alpha_bar_t = extract(
            self.alphas_cumprod, t, x_t
        )

        alphas_cumprod_prev = torch.cat([
            torch.ones(
                1,
                device=self.alphas_cumprod.device,
                dtype=self.alphas_cumprod.dtype
            ),
            self.alphas_cumprod[:-1]
        ])

        alpha_bar_prev = extract(
            alphas_cumprod_prev,
            t,
            x_t
        )

        # 1. noise prediction -> x0 prediction
        x0_pred = (
            x_t
            - torch.sqrt(1 - alpha_bar_t)
            * eps_theta
        ) / torch.sqrt(alpha_bar_t)

        x0_pred = x0_pred.clamp(-1, 1)

        # 2. true posterior mean
        coef1 = (
            torch.sqrt(alpha_bar_prev)
            * beta_t
            / (1 - alpha_bar_t)
        )

        coef2 = (
            torch.sqrt(alpha_t)
            * (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
        )

        posterior_mean = (
            coef1 * x0_pred
            + coef2 * x_t
        )

        # t=0: don't add noise
        if torch.all(t == 0):
            return posterior_mean

        # 3. posterior variance
        posterior_var = (
            (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
            * beta_t
        )

        noise = torch.randn_like(x_t)

        sample_prev = (
            posterior_mean
            + torch.sqrt(posterior_var) * noise
        )
        #######################
        return sample_prev

    
    def step_predict_x0(self, x_t: torch.Tensor, t: int, x0_pred: torch.Tensor):
        """
        x0 prediction version (alternative DDPM objective).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            x0_pred: predicted clean image x̂₀(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # Remember to clamp x0_pred to [-1, 1], as in step_predict_noise.

        beta_t = extract(
            self.betas, t, x_t
        )

        alpha_t = extract(
            self.alphas, t, x_t
        )

        alpha_bar_t = extract(
            self.alphas_cumprod, t, x_t
        )

        alphas_cumprod_prev = torch.cat([
            torch.ones(
                1,
                device=self.alphas_cumprod.device,
                dtype=self.alphas_cumprod.dtype
            ),
            self.alphas_cumprod[:-1]
        ])

        alpha_bar_prev = extract(
            alphas_cumprod_prev,
            t,
            x_t
        )

        x0_pred = x0_pred.clamp(-1, 1)

        coef1 = (
            torch.sqrt(alpha_bar_prev)
            * beta_t
            / (1 - alpha_bar_t)
        )

        coef2 = (
            torch.sqrt(alpha_t)
            * (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
        )

        posterior_mean = (
            coef1 * x0_pred
            + coef2 * x_t
        )

        if torch.all(t == 0):
            return posterior_mean

        posterior_var = (
            (1 - alpha_bar_prev)
            / (1 - alpha_bar_t)
            * beta_t
        )

        noise = torch.randn_like(x_t)

        sample_prev = (
            posterior_mean
            + torch.sqrt(posterior_var) * noise
        )

        #######################
        return sample_prev


    def step_predict_mean(self, x_t: torch.Tensor, t: int, mean_theta: torch.Tensor):
        """
        Mean prediction version (directly outputting the posterior mean).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            mean_theta: network-predicted posterior mean μ̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        if torch.all(t == 0):
            return mean_theta

        sigma_t = extract(
            self.sigmas,
            t,
            x_t
        )

        noise = torch.randn_like(x_t)

        sample_prev = (
            mean_theta
            + sigma_t * noise
        )
        #######################
        return sample_prev



    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        if eps is None:
            eps       = torch.randn(x_0.shape, device=x_0.device)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        alpha_bar_t = extract(
            self.alphas_cumprod,
            t,
            x_0
        )

        x_t = (
            torch.sqrt(alpha_bar_t) * x_0
            + torch.sqrt(1 - alpha_bar_t) * eps
        )
        #######################

        return x_t, eps
