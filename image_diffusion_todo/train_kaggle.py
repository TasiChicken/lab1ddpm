import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning import seed_everything
from tqdm import tqdm

from dataset import (
    AFHQDataModule,
    save_traj_strip,
    tensor_to_pil_image,
)
from model import DiffusionModule
from network import UNet
from scheduler import DDPMScheduler


# ============================================================
# Distributed utilities
# ============================================================

def setup_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.cuda.set_device(local_rank)

    distributed = world_size > 1

    if distributed:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    device = torch.device(
        f"cuda:{local_rank}"
    )

    return (
        distributed,
        rank,
        world_size,
        local_rank,
        device,
    )


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier(distributed):
    if distributed:
        dist.barrier()


# ============================================================
# Random states
# ============================================================

def set_training_seed(base_seed, rank):
    """
    Model initialization uses the same seed on all ranks.

    After model construction, each GPU gets a different random
    stream for DDPM timestep/noise sampling.
    """
    seed = base_seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(
            device
        ),
    }


def set_rng_state(state, device):
    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch.set_rng_state(
        state["torch_cpu"]
    )

    torch.cuda.set_rng_state(
        state["torch_cuda"],
        device=device,
    )


# ============================================================
# Safe file saving
# ============================================================

def atomic_torch_save(obj, path):
    """
    Write to .tmp first, then rename.

    If Kaggle unexpectedly terminates while saving, the
    previous valid checkpoint is less likely to be corrupted.
    """
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = Path(
        str(path) + ".tmp"
    )

    torch.save(
        obj,
        tmp_path,
    )

    os.replace(
        tmp_path,
        path,
    )


# ============================================================
# Resume checkpoint
# ============================================================

def rng_checkpoint_path(run_dir, rank):
    return (
        Path(run_dir)
        / f"resume_rank{rank}_rng.pt"
    )


def save_resume_checkpoint(
    run_dir,
    raw_network,
    optimizer,
    lr_scheduler,
    step,
    epoch,
    batch_in_epoch,
    last_logged_step,
    losses,
    args,
    distributed,
    rank,
    world_size,
    device,
):
    """
    Exact-resume checkpoint.

    Every rank stores its own RNG state.
    Rank 0 stores model/optimizer/LR scheduler/global state.
    """

    run_dir = Path(run_dir)

    # Each GPU has a different DDPM random stream.
    atomic_torch_save(
        get_rng_state(device),
        rng_checkpoint_path(
            run_dir,
            rank,
        ),
    )

    barrier(distributed)

    if rank == 0:
        checkpoint = {
            "version": 1,

            # Training position
            "step": step,
            "epoch": epoch,
            "batch_in_epoch":
                batch_in_epoch,

            "last_logged_step":
                last_logged_step,

            # Training states
            "model_state_dict":
                raw_network.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "lr_scheduler_state_dict":
                lr_scheduler.state_dict(),

            # Logging
            "losses": losses,

            # Resume validation
            "world_size": world_size,

            "global_batch_size":
                args.batch_size,

            "mode": args.mode,

            "predictor":
                args.predictor,

            "num_diffusion_train_timesteps":
                args.num_diffusion_train_timesteps,

            "warmup_steps":
                args.warmup_steps,

            "learning_rate":
                args.lr,
        }

        atomic_torch_save(
            checkpoint,
            run_dir / "resume.pt",
        )

    barrier(distributed)


def load_resume_checkpoint(
    resume_path,
    raw_network,
    optimizer,
    lr_scheduler,
    args,
    world_size,
    device,
):
    checkpoint = torch.load(
        resume_path,
        map_location=device,
        weights_only=False,
    )

    # --------------------------------------------------------
    # Make sure experiment settings still match.
    # --------------------------------------------------------

    if checkpoint["world_size"] != world_size:
        raise ValueError(
            "Resume world_size mismatch: "
            f"checkpoint={checkpoint['world_size']}, "
            f"current={world_size}"
        )

    if (
        checkpoint["global_batch_size"]
        != args.batch_size
    ):
        raise ValueError(
            "Resume batch size mismatch: "
            f"checkpoint="
            f"{checkpoint['global_batch_size']}, "
            f"current={args.batch_size}"
        )

    if checkpoint["mode"] != args.mode:
        raise ValueError(
            "Resume scheduler mismatch: "
            f"{checkpoint['mode']} "
            f"!= {args.mode}"
        )

    if (
        checkpoint["predictor"]
        != args.predictor
    ):
        raise ValueError(
            "Resume predictor mismatch: "
            f"{checkpoint['predictor']} "
            f"!= {args.predictor}"
        )

    if (
        checkpoint[
            "num_diffusion_train_timesteps"
        ]
        != args.num_diffusion_train_timesteps
    ):
        raise ValueError(
            "Diffusion timestep mismatch."
        )

    if (
        checkpoint["warmup_steps"]
        != args.warmup_steps
    ):
        raise ValueError(
            "Warmup step mismatch."
        )

    if (
        checkpoint["learning_rate"]
        != args.lr
    ):
        raise ValueError(
            "Learning rate mismatch."
        )

    # --------------------------------------------------------
    # Restore exact training states.
    # --------------------------------------------------------

    raw_network.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    lr_scheduler.load_state_dict(
        checkpoint[
            "lr_scheduler_state_dict"
        ]
    )

    return {
        "step":
            int(checkpoint["step"]),

        "epoch":
            int(checkpoint["epoch"]),

        "batch_in_epoch":
            int(
                checkpoint[
                    "batch_in_epoch"
                ]
            ),

        "last_logged_step":
            int(
                checkpoint.get(
                    "last_logged_step",
                    -1,
                )
            ),

        "losses":
            checkpoint.get(
                "losses",
                [],
            ),
    }


# ============================================================
# Old starter-code checkpoint support
# ============================================================

def load_old_assignment_checkpoint(
    raw_network,
    ckpt_path,
    args,
):
    """
    Load an old last.ckpt produced by the original train.py.

    IMPORTANT:
    Original last.ckpt contains model weights but NOT:
        - Adam optimizer state
        - LambdaLR state
        - training step

    Therefore this is NOT exact resume.

    Use only to salvage an existing long training run.
    """

    checkpoint = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    hparams = checkpoint.get(
        "hparams",
        {},
    )

    old_predictor = hparams.get(
        "predictor",
        None,
    )

    if (
        old_predictor is not None
        and old_predictor != args.predictor
    ):
        raise ValueError(
            "Old checkpoint predictor "
            f"{old_predictor} != "
            f"{args.predictor}"
        )

    old_scheduler = hparams.get(
        "var_scheduler",
        None,
    )

    if old_scheduler is not None:
        old_mode = getattr(
            old_scheduler,
            "schedule_mode",
            None,
        )

        if (
            old_mode is not None
            and old_mode != args.mode
        ):
            raise ValueError(
                "Old checkpoint scheduler "
                f"{old_mode} != "
                f"{args.mode}"
            )

    state_dict = checkpoint[
        "state_dict"
    ]

    network_state = {}

    for key, value in state_dict.items():
        if key.startswith("network."):
            new_key = key[
                len("network.") :
            ]

            network_state[
                new_key
            ] = value

    raw_network.load_state_dict(
        network_state,
        strict=True,
    )

    print(
        f"Loaded old model weights: "
        f"{ckpt_path}"
    )


def align_fresh_scheduler_to_step(
    optimizer,
    lr_scheduler,
    completed_steps,
    base_lr,
    warmup_steps,
):
    """
    Used only for old starter-code checkpoint salvage.

    Adam is fresh because the old checkpoint did not store its
    state, but LR is aligned to the corresponding global step.
    """

    if completed_steps < 0:
        raise ValueError(
            "completed_steps must be >= 0"
        )

    factor = min(
        (completed_steps + 1)
        / warmup_steps,
        1.0,
    )

    target_lr = (
        base_lr * factor
    )

    for param_group in (
        optimizer.param_groups
    ):
        param_group[
            "lr"
        ] = target_lr

    state = (
        lr_scheduler.state_dict()
    )

    state[
        "last_epoch"
    ] = completed_steps

    state[
        "_step_count"
    ] = completed_steps + 1

    state[
        "_last_lr"
    ] = [
        target_lr
        for _ in optimizer.param_groups
    ]

    lr_scheduler.load_state_dict(
        state
    )


# ============================================================
# Assignment-compatible checkpoint
# ============================================================

def save_assignment_checkpoint(
    file_path,
    raw_network,
    var_scheduler,
    predictor,
):
    """
    Produce the same checkpoint format expected by the existing
    sampling.py.
    """

    ddpm_for_save = DiffusionModule(
        raw_network,
        var_scheduler,
        predictor=predictor,
    )

    ddpm_for_save.save(
        file_path
    )


def save_milestone_checkpoint(
    run_dir,
    step,
    raw_network,
    var_scheduler,
    predictor,
):
    """
    Save an assignment-compatible checkpoint for FID tuning.

    Example:
        step_5000.ckpt
        step_10000.ckpt
        ...
    """
    run_dir = Path(run_dir)

    save_assignment_checkpoint(
        run_dir / f"step_{step}.ckpt",
        raw_network,
        var_scheduler,
        predictor,
    )


# ============================================================
# Loss figure
# ============================================================

def save_loss_figure(
    losses,
    path,
):
    plt.figure()

    plt.plot(
        losses
    )

    plt.title(
        "Loss curve"
    )

    plt.xlabel(
        "Training step"
    )

    plt.ylabel(
        "Loss"
    )

    plt.tight_layout()

    plt.savefig(
        path
    )

    plt.close()


# ============================================================
# Sampling / logging
# ============================================================

@torch.no_grad()
def perform_original_style_logging(
    run_dir,
    step,
    losses,
    raw_network,
    var_scheduler,
    predictor,
):
    """
    Mirrors the original train.py logging behavior:

        ddpm.eval()
        save loss.png
        generate 4 samples
        generate one trajectory
        save last.ckpt
        ddpm.train()

    Only rank 0 calls this function.
    """

    run_dir = Path(run_dir)

    raw_network.eval()

    save_loss_figure(
        losses,
        run_dir / "loss.png",
    )

    sampling_ddpm = DiffusionModule(
        raw_network,
        var_scheduler,
        predictor=predictor,
    )

    # --------------------------------------------------------
    # Same as original train.py:
    # samples = ddpm.sample(4)
    # --------------------------------------------------------

    samples = sampling_ddpm.sample(
        4,
        return_traj=False,
    )

    pil_images = tensor_to_pil_image(
        samples
    )

    for i, img in enumerate(
        pil_images
    ):
        img.save(
            run_dir
            / f"step={step}-{i}.png"
        )

    # --------------------------------------------------------
    # Same trajectory behavior as original train.py.
    # --------------------------------------------------------

    traj = sampling_ddpm.sample(
        1,
        return_traj=True,
    )

    save_traj_strip(
        run_dir
        / f"step={step}-traj.png",

        traj,

        num_frames=10,

        pad=4,
    )

    # --------------------------------------------------------
    # Same assignment checkpoint format.
    # --------------------------------------------------------

    save_assignment_checkpoint(
        run_dir / "last.ckpt",
        raw_network,
        var_scheduler,
        predictor,
    )

    raw_network.train()


# ============================================================
# Data iterator helpers
# ============================================================

def make_iterator_at_position(
    train_loader,
    train_sampler,
    epoch,
    batch_in_epoch,
):
    if train_sampler is not None:
        train_sampler.set_epoch(
            epoch
        )

    train_iter = iter(
        train_loader
    )

    # Exact-resume data position.
    for _ in range(
        batch_in_epoch
    ):
        try:
            next(train_iter)
        except StopIteration:
            raise RuntimeError(
                "Resume batch position "
                "is outside this epoch."
            )

    return train_iter


# ============================================================
# Main
# ============================================================

def main(args):
    (
        distributed,
        rank,
        world_size,
        local_rank,
        device,
    ) = setup_distributed()

    is_main = rank == 0

    run_dir = Path(
        args.run_dir
    )

    if is_main:
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    barrier(distributed)

    # --------------------------------------------------------
    # IMPORTANT:
    # Same initialization seed as original starter code.
    # --------------------------------------------------------

    seed_everything(
        args.seed
    )

    # Optional speed optimization.
    # Does not change the training objective.
    torch.backends.cudnn.benchmark = True

    # ========================================================
    # Dataset
    # ========================================================

    # Allow rank 0 to download/build dataset first.
    if is_main:
        ds_module = AFHQDataModule(
            "./data",

            batch_size=
                args.batch_size,

            num_workers=4,

            max_num_images_per_cat=
                args.max_num_images_per_cat,

            image_resolution=64,
        )

    barrier(distributed)

    if not is_main:
        ds_module = AFHQDataModule(
            "./data",

            batch_size=
                args.batch_size,

            num_workers=4,

            max_num_images_per_cat=
                args.max_num_images_per_cat,

            image_resolution=64,
        )

    if (
        args.batch_size
        % world_size
        != 0
    ):
        raise ValueError(
            "Global batch size must be "
            "divisible by number of GPUs. "
            f"batch={args.batch_size}, "
            f"world_size={world_size}"
        )

    local_batch_size = (
        args.batch_size
        // world_size
    )

    if distributed:
        train_sampler = (
            DistributedSampler(
                ds_module.train_ds,

                num_replicas=
                    world_size,

                rank=rank,

                shuffle=True,

                seed=args.seed,

                drop_last=True,
            )
        )
    else:
        train_sampler = None

    # With 2 GPUs and global batch 16:
    #     local batch = 8
    #
    # With 2 GPUs and global batch 32:
    #     local batch = 16
    #
    # num_workers=2 per GPU => total 4 workers,
    # matching the original train.py's total worker count.
    train_loader = DataLoader(
        ds_module.train_ds,

        batch_size=
            local_batch_size,

        sampler=
            train_sampler,

        shuffle=(
            train_sampler is None
        ),

        num_workers=
            args.num_workers_per_gpu,

        pin_memory=True,

        persistent_workers=(
            args.num_workers_per_gpu
            > 0
        ),

        drop_last=True,
    )

    # ========================================================
    # SAME scheduler as original train.py
    # ========================================================

    var_scheduler = DDPMScheduler(
        args.num_diffusion_train_timesteps,

        beta_1=
            args.beta_1,

        beta_T=
            args.beta_T,

        mode=
            args.mode,
    )

    # ========================================================
    # SAME UNet as original train.py
    # ========================================================

    raw_network = UNet(
        T=
            args.num_diffusion_train_timesteps,

        image_resolution=64,

        ch=128,

        ch_mult=[
            1, 2, 2, 2
        ],

        attn=[1],

        num_res_blocks=4,

        dropout=0.1,

        use_cfg=False,

        cfg_dropout=0.1,

        num_classes=getattr(
            ds_module,
            "num_classes",
            None,
        ),
    )

    # --------------------------------------------------------
    # Optional salvage of old original train.py checkpoint.
    # --------------------------------------------------------

    if (
        args.init_ckpt
        and not args.resume
    ):
        load_old_assignment_checkpoint(
            raw_network,
            args.init_ckpt,
            args,
        )

    raw_network = raw_network.to(
        device
    )

    var_scheduler = (
        var_scheduler.to(
            device
        )
    )

    # --------------------------------------------------------
    # DDP synchronizes one model across both GPUs.
    # --------------------------------------------------------

    if distributed:
        network = DDP(
            raw_network,

            device_ids=[
                local_rank
            ],

            output_device=
                local_rank,

            broadcast_buffers=True,

            find_unused_parameters=False,
        )
    else:
        network = raw_network

    # ========================================================
    # SAME DiffusionModule as original train.py
    # ========================================================

    ddpm = DiffusionModule(
        network,
        var_scheduler,
        predictor=args.predictor,
    )

    # ========================================================
    # SAME Adam optimizer as original train.py
    # ========================================================

    optimizer = torch.optim.Adam(
        ddpm.network.parameters(),
        lr=args.lr,
    )

    # ========================================================
    # SAME LambdaLR formula as original train.py
    # ========================================================

    lr_scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,

            lr_lambda=lambda t:
                min(
                    (t + 1)
                    / args.warmup_steps,
                    1.0,
                ),
        )
    )

    # ========================================================
    # Initial training state
    # ========================================================

    step = 0
    epoch = 0
    batch_in_epoch = 0

    losses = []

    last_logged_step = -1

    resume_state = None

    # ========================================================
    # Exact resume
    # ========================================================

    if args.resume:
        resume_state = (
            load_resume_checkpoint(
                args.resume,

                raw_network,

                optimizer,

                lr_scheduler,

                args,

                world_size,

                device,
            )
        )

        step = (
            resume_state["step"]
        )

        epoch = (
            resume_state["epoch"]
        )

        batch_in_epoch = (
            resume_state[
                "batch_in_epoch"
            ]
        )

        last_logged_step = (
            resume_state[
                "last_logged_step"
            ]
        )

        losses = (
            resume_state[
                "losses"
            ]
        )

        if is_main:
            print(
                "\n"
                "================================"
            )

            print(
                "Exact resume loaded"
            )

            print(
                f"step = {step}"
            )

            print(
                f"epoch = {epoch}"
            )

            print(
                "batch_in_epoch = "
                f"{batch_in_epoch}"
            )

            print(
                "learning rate = "
                f"{optimizer.param_groups[0]['lr']:.8f}"
            )

            print(
                "================================"
                "\n"
            )

    # ========================================================
    # Old checkpoint salvage
    # ========================================================

    elif args.init_ckpt:
        step = args.start_step

        # Infer approximate data position.
        batches_per_epoch = len(
            train_loader
        )

        epoch = (
            step
            // batches_per_epoch
        )

        batch_in_epoch = (
            step
            % batches_per_epoch
        )

        align_fresh_scheduler_to_step(
            optimizer,
            lr_scheduler,
            step,
            args.lr,
            args.warmup_steps,
        )

        if is_main:
            print(
                "\nWARNING:"
            )

            print(
                "Old last.ckpt contains "
                "model weights only."
            )

            print(
                "Adam state cannot be "
                "recovered."
            )

            print(
                "This is weight-only "
                "continuation, NOT exact resume."
            )

            print(
                f"Starting global step: "
                f"{step}\n"
            )

    # --------------------------------------------------------
    # Give each GPU an independent DDPM random stream.
    #
    # For exact resume this will later be replaced with the
    # saved rank-specific RNG state.
    # --------------------------------------------------------

    set_training_seed(
        args.seed,
        rank,
    )

    # ========================================================
    # Restore data position
    # ========================================================

    train_iter = (
        make_iterator_at_position(
            train_loader,

            train_sampler,

            epoch,

            batch_in_epoch,
        )
    )

    # ========================================================
    # Restore exact per-rank random state AFTER rebuilding
    # the DataLoader iterator / skipping old batches.
    # ========================================================

    if args.resume:
        rng_file = rng_checkpoint_path(
            run_dir,
            rank,
        )

        if not rng_file.exists():
            raise FileNotFoundError(
                "Missing rank-specific RNG "
                f"checkpoint: {rng_file}"
            )

        rank_rng_state = torch.load(
            rng_file,
            map_location="cpu",
            weights_only=False,
        )

        set_rng_state(
            rank_rng_state,
            device,
        )

    # ========================================================
    # Save runtime configuration
    # ========================================================

    if is_main:
        runtime_config = dict(
            vars(args)
        )

        runtime_config[
            "world_size"
        ] = world_size

        runtime_config[
            "local_batch_size"
        ] = local_batch_size

        runtime_config[
            "effective_global_batch_size"
        ] = (
            local_batch_size
            * world_size
        )

        with open(
            run_dir
            / "config.json",

            "w",
        ) as f:
            json.dump(
                runtime_config,
                f,
                indent=2,
            )

        print(
            "================================"
        )

        print(
            f"GPUs: {world_size}"
        )

        print(
            "Global batch size: "
            f"{args.batch_size}"
        )

        print(
            "Local batch size/GPU: "
            f"{local_batch_size}"
        )

        print(
            f"Predictor: "
            f"{args.predictor}"
        )

        print(
            f"Beta schedule: "
            f"{args.mode}"
        )

        print(
            f"Target steps: "
            f"{args.train_num_steps}"
        )

        print(
            "================================"
        )

    # ========================================================
    # Progress bar only on rank 0
    # ========================================================

    if is_main:
        pbar = tqdm(
            total=
                args.train_num_steps,

            initial=step,
        )
    else:
        pbar = None

    session_start_time = (
        time.time()
    )

    stopped_for_time = False

    # ========================================================
    # Training loop
    # ========================================================

    while (
        step
        < args.train_num_steps
    ):
        # ----------------------------------------------------
        # Kaggle time-limit check.
        #
        # Check every N steps, not every step, to avoid adding
        # an unnecessary distributed collective each update.
        # ----------------------------------------------------

        if (
            args.max_hours > 0
            and (
                step
                % args.time_check_interval
                == 0
            )
        ):
            elapsed_hours = (
                time.time()
                - session_start_time
            ) / 3600.0

            local_stop = (
                1
                if elapsed_hours
                >= args.max_hours
                else 0
            )

            stop_tensor = torch.tensor(
                [local_stop],

                dtype=torch.int32,

                device=device,
            )

            if distributed:
                dist.all_reduce(
                    stop_tensor,
                    op=dist.ReduceOp.MAX,
                )

            if (
                stop_tensor.item()
                != 0
            ):
                if is_main:
                    print(
                        "\nReached Kaggle "
                        "session safety limit."
                    )

                    print(
                        f"Saving at step "
                        f"{step}..."
                    )

                save_resume_checkpoint(
                    run_dir,

                    raw_network,

                    optimizer,

                    lr_scheduler,

                    step,

                    epoch,

                    batch_in_epoch,

                    last_logged_step,

                    losses,

                    args,

                    distributed,

                    rank,

                    world_size,

                    device,
                )

                stopped_for_time = True

                break

        # ====================================================
        # SAME original train.py logging block
        # ====================================================

        if (
            args.log_interval > 0
            and step
                % args.log_interval
                == 0
            and step
                != last_logged_step
        ):
            # Rank 1 waits while rank 0 performs expensive
            # DDPM reverse sampling.
            barrier(distributed)

            if is_main:
                perform_original_style_logging(
                    run_dir,

                    step,

                    losses,

                    raw_network,

                    var_scheduler,

                    args.predictor,
                )

            barrier(distributed)

            # Keep the bookkeeping identical on all ranks.
            last_logged_step = step

        # ====================================================
        # Get next batch
        # ====================================================

        try:
            img, label = next(
                train_iter
            )

        except StopIteration:
            epoch += 1

            batch_in_epoch = 0

            train_iter = (
                make_iterator_at_position(
                    train_loader,

                    train_sampler,

                    epoch,

                    batch_in_epoch,
                )
            )

            img, label = next(
                train_iter
            )

        img = img.to(
            device,
            non_blocking=True,
        )

        # label intentionally unused:
        # this is the same unconditional Lab-1 training flow.

        # ====================================================
        # SAME training order as original train.py
        # ====================================================

        # Original:
        #   loss = ddpm.get_loss(img)

        loss = ddpm.get_loss(
            img
        )

        # For progress display only.
        loss_for_log = (
            loss.detach().clone()
        )

        if distributed:
            dist.all_reduce(
                loss_for_log,
                op=dist.ReduceOp.SUM,
            )

            loss_for_log /= (
                world_size
            )

        # Original:
        #   optimizer.zero_grad()
        optimizer.zero_grad()

        # Original:
        #   loss.backward()
        loss.backward()

        # DDP automatically synchronizes/averages gradients
        # during backward().

        # Original:
        #   optimizer.step()
        optimizer.step()

        # Original:
        #   scheduler.step()
        lr_scheduler.step()

        # ----------------------------------------------------
        # Same semantic step count as original train.py:
        # one global optimizer update = one step.
        # ----------------------------------------------------

        step += 1

        batch_in_epoch += 1

        # If this batch completed the current epoch,
        # mark position at the beginning of next epoch.
        if (
            batch_in_epoch
            >= len(train_loader)
        ):
            epoch += 1

            batch_in_epoch = 0

            train_iter = (
                make_iterator_at_position(
                    train_loader,

                    train_sampler,

                    epoch,

                    0,
                )
            )

        # ----------------------------------------------------
        # Original losses.append(loss.item()).
        #
        # Under DDP, rank-0 stores the mean of both GPU losses,
        # corresponding to the global batch.
        # ----------------------------------------------------

        if is_main:
            current_loss = (
                loss_for_log.item()
            )

            losses.append(
                current_loss
            )

            current_lr = (
                optimizer
                .param_groups[0]
                ["lr"]
            )

            pbar.set_description(
                "Loss: "
                f"{current_loss:.4f} "
                "| "
                "LR: "
                f"{current_lr:.2e}"
            )

            pbar.update(1)

        # ====================================================
        # FID milestone checkpoint
        # ====================================================

        if (
            args.milestone_interval > 0
            and step % args.milestone_interval == 0
        ):
            barrier(distributed)

            if is_main:
                save_milestone_checkpoint(
                    run_dir,
                    step,
                    raw_network,
                    var_scheduler,
                    args.predictor,
                )

                print(
                    "\nFID milestone checkpoint saved "
                    f"at step {step}: "
                    f"{run_dir / f'step_{step}.ckpt'}"
                )

            barrier(distributed)

        # ====================================================
        # Exact-resume checkpoint
        # ====================================================

        if (
            args.resume_interval > 0
            and step
                % args.resume_interval
                == 0
        ):
            save_resume_checkpoint(
                run_dir,

                raw_network,

                optimizer,

                lr_scheduler,

                step,

                epoch,

                batch_in_epoch,

                last_logged_step,

                losses,

                args,

                distributed,

                rank,

                world_size,

                device,
            )

            if is_main:
                print(
                    "\nExact resume "
                    f"checkpoint saved "
                    f"at step {step}"
                )

    # ========================================================
    # End training
    # ========================================================

    if (
        step
        >= args.train_num_steps
    ):
        barrier(distributed)

        if is_main:
            print(
                "\nTraining completed."
            )

            save_loss_figure(
                losses,
                run_dir / "loss.png",
            )

            # Same format as original train.py,
            # usable directly by sampling.py.
            save_assignment_checkpoint(
                run_dir / "last.ckpt",

                raw_network,

                var_scheduler,

                args.predictor,
            )


            # Also preserve the final model as a numbered
            # assignment-compatible checkpoint for FID comparison.
            final_milestone_path = (
                run_dir / f"step_{step}.ckpt"
            )

            if not final_milestone_path.exists():
                save_milestone_checkpoint(
                    run_dir,
                    step,
                    raw_network,
                    var_scheduler,
                    args.predictor,
                )

        barrier(distributed)

        # Also save exact final resume state.
        save_resume_checkpoint(
            run_dir,

            raw_network,

            optimizer,

            lr_scheduler,

            step,

            epoch,

            batch_in_epoch,

            last_logged_step,

            losses,

            args,

            distributed,

            rank,

            world_size,

            device,
        )

        if is_main:
            print(
                "Final files:"
            )

            print(
                run_dir / "last.ckpt"
            )

            print(
                run_dir / "resume.pt"
            )

    elif stopped_for_time:
        if is_main:
            print(
                "\nSession ended safely."
            )

            print(
                "Resume next session with:"
            )

            print(
                f"--resume "
                f"{run_dir / 'resume.pt'}"
            )

    if pbar is not None:
        pbar.close()

    cleanup_distributed()


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # Same experiment parameters as starter train.py
    # --------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help=(
            "GLOBAL batch size across all GPUs. "
            "With 2 GPUs, batch 32 means 16/GPU."
        ),
    )

    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    parser.add_argument(
        "--max_num_images_per_cat",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--beta_1",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--beta_T",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=63,
    )

    parser.add_argument(
        "--predictor",
        type=str,
        default="noise",
        choices=[
            "noise",
            "x0",
            "mean",
        ],
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="linear",
        choices=[
            "linear",
            "cosine",
            "quad",
        ],
    )

    # --------------------------------------------------------
    # Kaggle/DDP infrastructure
    # --------------------------------------------------------

    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--num_workers_per_gpu",
        type=int,
        default=2,
        help=(
            "2 workers/GPU on 2 GPUs gives 4 workers total, "
            "matching the starter code's total worker count."
        ),
    )

    # Same logging functionality as original train.py.
    # Default remains 200 like starter code.
    # For Kaggle long runs you can explicitly use 5000.
    parser.add_argument(
        "--log_interval",
        type=int,
        default=200,
        help=(
            "Original-style image/trajectory logging interval. "
            "Use 0 to disable during correctness tests."
        ),
    )

    parser.add_argument(
        "--milestone_interval",
        type=int,
        default=0,
        help=(
            "Save numbered assignment-compatible checkpoints "
            "(step_5000.ckpt, etc.) every N steps for FID tuning. "
            "Use 0 to disable."
        ),
    )

    # Exact resume checkpoint can be more frequent than expensive
    # DDPM sampling.
    parser.add_argument(
        "--resume_interval",
        type=int,
        default=1000,
    )

    # Stop safely before Kaggle hard termination.
    # 0 disables automatic stop.
    parser.add_argument(
        "--max_hours",
        type=float,
        default=10.5,
    )

    parser.add_argument(
        "--time_check_interval",
        type=int,
        default=100,
    )

    # --------------------------------------------------------
    # Exact resume created by THIS script.
    # --------------------------------------------------------

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Path to resume.pt generated by train_kaggle.py."
        ),
    )

    # --------------------------------------------------------
    # Optional salvage of OLD starter train.py checkpoint.
    # NOT exact resume.
    # --------------------------------------------------------

    parser.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help=(
            "Old last.ckpt from starter train.py. "
            "Loads model weights only."
        ),
    )

    parser.add_argument(
        "--start_step",
        type=int,
        default=0,
        help=(
            "Completed step count represented by --init_ckpt."
        ),
    )

    args = parser.parse_args()

    if (
        args.resume
        and args.init_ckpt
    ):
        parser.error(
            "Use either --resume or --init_ckpt, not both."
        )

    if (
        args.start_step > 0
        and not args.init_ckpt
    ):
        parser.error(
            "--start_step is only for --init_ckpt."
        )

    main(args)