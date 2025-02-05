# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from datetime import timedelta

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.fx import GraphModule
from torchtitan.checkpoint import CheckpointManager, TrainState
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_gpu_memory_monitor, build_metric_logger
from torchtitan.models import (
    model_name_to_cls,
    model_name_to_tokenizer,
    model_name_to_weights_download_fns,
    model_name_to_weights_export_fns,
    models_config,
)
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import models_parallelize_fns, ParallelDims
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from torchtitan.tokenizers.tokenizer import build_tokenizer

from torchtitan.utils import common_utils as utils
from torchtitan.validation import validate


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger(job_config.logging.log_level)
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = utils.Color if job_config.metrics.enable_color_printing else utils.NoColor

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=job_config.training.data_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
        dp_type=job_config.training.data_parallel_type,
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    utils.init_distributed(job_config)
    # initialize GPU memory monitor and get peak flops for MFU calculation
    gpu_memory_monitor = build_gpu_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(gpu_memory_monitor.device_name)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    model_name = job_config.model.name
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    init_device = "cpu" if job_config.checkpoint.create_seed_checkpoint else "cuda"

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build training dataloader
    representation_type = job_config.training.representation_type
    data_loader = build_hf_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_path,
        job_config.training.data_processing_style,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
        representation_type,
        pin_memory=job_config.dataloader.pin_memory,
        num_workers=job_config.dataloader.num_workers,
        special_mode=job_config.dataloader.special_mode,
    )

    if not job_config.validation.batch_size:
        logger.info("Validation batch size not specified, training batch size is used.")
        job_config.validation.batch_size = job_config.training.batch_size

    if job_config.validation.enable_valid:
        # build validation dataloader
        valid_data_loader = build_hf_data_loader(
            job_config.validation.dataset,
            job_config.validation.dataset_path,
            job_config.training.data_processing_style,
            tokenizer,
            job_config.validation.batch_size,
            job_config.training.seq_len,
            dp_degree,
            dp_rank,
            representation_type,
            infinite=False,
            pin_memory=job_config.dataloader.pin_memory,
            num_workers=job_config.dataloader.num_workers,
            special_mode=job_config.dataloader.special_mode,
        )

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.padded_n_words
    model_config.max_seq_len = job_config.training.seq_len

    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    # load the model on rank 0 only, then FSDP will distribute the weights
    if job_config.model_download_export.to_titan:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        model.to_empty(device=init_device)
        model_name_to_weights_download_fns[model_name](
            model,
            weights_path=job_config.checkpoint.load_folder,
            tokenizer=tokenizer.model,
            source=job_config.model_download_export.weights_source,
            token_embedding_size=model_config.vocab_size,
        )

    # a no-op hander if float8 is not enabled
    float8_handler = Float8Handler(job_config, parallel_dims)
    # swap to Float8Linear based on float8 configs
    float8_handler.convert_to_float8_training(model)

    # log model size
    model_param_count = utils.get_num_params(model)

    num_flop_per_token = utils.get_num_flop_per_token(
        utils.get_num_params(model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )

    num_flop_per_token_val = utils.get_num_flop_per_token_forward(
        utils.get_num_params(model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )

    logger.info(
        f"{color.blue}Model {model_name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # loss function to be shared by Pipeline Parallel and SPMD training
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1), labels.flatten(0, 1)
        )

    # apply parallelisms and initialization
    # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
    models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if not job_config.model_download_export.to_titan:
        model.to_empty(device=init_device)
    model_parts = [model]

    for mod in model_parts:
        # skip traced modules since we do not define init_weights in the traced module
        if isinstance(mod, GraphModule):
            continue
        if not job_config.model_download_export.to_titan:
            mod.init_weights()
        mod.train()

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = build_optimizers(model_parts, job_config)
    lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)

    train_state = TrainState()

    metric_logger = build_metric_logger(job_config, parallel_dims)
    metric_logger.log_hparams(job_config.args_dict)

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers.optimizers,
        lr_schedulers=lr_schedulers.schedulers,
        states={"train_state": train_state},
        job_config=job_config,
        experiment_hash=metric_logger.experiment_hash,
    )

    if job_config.model_download_export.to_titan:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created titan checkpoint")
        return

    checkpoint_loaded = checkpoint.load(job_config.checkpoint.load_at_step)

    if job_config.model_download_export.to_hf:
        assert (
            world_size == 1
        ), "Must create seed-checkpoint using one gpu, to disable sharding"
        model_name_to_weights_export_fns[model_name](
            model,
            save_dir=checkpoint._create_checkpoint_id(
                job_config.checkpoint.load_at_step, checkpoint.save_folder
            ),
            tokenizer=tokenizer.model,
            token_embedding_size=model_config.vocab_size,
        )
        logger.info("Created huggingface checkpoint")
        return

    data_iterator = iter(data_loader)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # variables used to keep info for metrics logging
    losses_since_last_log = []
    ntokens_since_last_log = 0
    data_loading_times = []
    time_last_log = time.perf_counter()
    gpu_memory_monitor.reset_peak_stats()

    checkpoint.reset()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size * job_config.training.gradient_accumulation_steps}, "
        f"global batch size {job_config.training.batch_size * job_config.training.gradient_accumulation_steps * dp_degree}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        logger.debug("Got into profiling context")
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            optimizers.zero_grad()
            logger.debug("step")

            loss = 0
            for _ in range(job_config.training.gradient_accumulation_steps):
                batch = next(data_iterator, None)
                input_ids, labels = batch

                ntokens_since_last_log += labels.numel()
                input_ids = input_ids.cuda()
                labels = labels.cuda()
                data_loading_times.append(time.perf_counter() - data_load_start)

                with train_context():
                    logger.debug("enter context")
                    pred = model(input_ids)
                    cur_loss = loss_fn(pred, labels)
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del pred
                    cur_loss /= job_config.training.gradient_accumulation_steps
                    cur_loss.backward()
                    loss += cur_loss.detach().clone()

                for m in model_parts:
                    torch.nn.utils.clip_grad_norm_(
                        m.parameters(), job_config.training.max_norm, foreach=True
                    )

            # sync float8 amaxes and scales
            float8_handler.sync_float8_amax_and_scale_history(model_parts)

            checkpoint.maybe_wait_for_staging()
            # optimizer step
            optimizers.step()
            lr_schedulers.step()

            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model_parts)

            # loss /= job_config.training.gradient_accumulation_steps
            losses_since_last_log.append(loss)

            # log train metrics
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                losses = [loss.item() for loss in losses_since_last_log]

                perplexities = [2 ** loss.item() for loss in losses_since_last_log]

                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                avg_perplexity, max_perplexity = (
                    sum(perplexities) / len(perplexities),
                    max(perplexities),
                )

                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(avg_loss, dp_mesh),
                        utils.dist_max(max_loss, dp_mesh),
                    )
                    global_avg_perplexity, global_max_perplexity = (
                        utils.dist_mean(avg_perplexity, dp_mesh),
                        utils.dist_max(max_perplexity, dp_mesh),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss
                    global_avg_perplexity, global_max_perplexity = (
                        avg_perplexity,
                        max_perplexity,
                    )

                # update train state
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)
                train_state.global_avg_perplexities.append(global_avg_perplexity)
                train_state.global_max_perplexities.append(global_max_perplexity)

                time_delta = time.perf_counter() - time_last_log

                # tokens per second, abbr. as wps by convention
                wps = ntokens_since_last_log / (
                    time_delta * parallel_dims.model_parallel_size
                )
                # model FLOPS utilization
                # For its definition and calculation, please refer to the PaLM paper:
                # https://arxiv.org/abs/2204.02311
                mfu = 100 * num_flop_per_token * wps / gpu_peak_flops

                time_end_to_end = time_delta / job_config.metrics.log_freq
                time_data_loading = sum(data_loading_times) / len(data_loading_times)
                time_data_loading_pct = 100 * sum(data_loading_times) / time_delta

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

                metrics = {
                    "train/loss_metrics/global_avg_loss": global_avg_loss,
                    "train/loss_metrics/global_max_loss": global_max_loss,
                    "train/loss_metrics/global_avg_perplexity": global_avg_perplexity,
                    "train/loss_metrics/global_max_perplexity": global_max_perplexity,
                    "train/wps": wps,
                    "train/mfu(%)": mfu,
                    "lr": lr_schedulers.last_lr,
                    "train/time_metrics/end_to_end(s)": time_end_to_end,
                    "train/time_metrics/data_loading(s)": time_data_loading,
                    "train/time_metrics/data_loading(%)": time_data_loading_pct,
                    "train/memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
                    "train/memory/max_active(%)": gpu_mem_stats.max_active_pct,
                    "train/memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
                    "train/memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
                    "train/memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
                    "train/memory/num_ooms": gpu_mem_stats.num_ooms,
                }
                metric_logger.log(metrics, step=train_state.step)

                logger.info(
                    "context: train  "
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}wps: {round(wps):,}  "
                    f"{color.magenta}mfu: {mfu:.2f}%{color.reset}  "
                    f"{color.red}lr: {lr_schedulers.last_lr:.3e}{color.reset}"
                )

                losses_since_last_log.clear()
                ntokens_since_last_log = 0
                data_loading_times.clear()
                time_last_log = time.perf_counter()
                gpu_memory_monitor.reset_peak_stats()

            # log val metrics
            if job_config.validation.enable_valid and (
                train_state.step == 0
                or train_state.step % job_config.validation.valid_freq == 0
            ):
                validate(
                    model,
                    valid_data_loader,
                    logger,
                    metric_logger,
                    parallel_dims,
                    gc_handler,
                    gpu_memory_monitor,
                    color,
                    train_state.step,
                    num_flop_per_token_val,
                    gpu_peak_flops,
                    dp_rank,
                    dp_mesh,
                    world_size,
                    job_config.experimental.enable_compiled_autograd,
                    device,
                )

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
