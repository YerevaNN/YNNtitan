import gc
import os
import time
from logging import Logger
from typing import Any, List, Union

import torch
import torch.distributed as dist
from torch.nn.functional import cross_entropy

from torchtitan.checkpoint import TrainState
from torchtitan.datasets.hf_datasets import DPAwareDataLoader
from torchtitan.metrics import GPUMemoryMonitor, MetricLogger
from torchtitan.parallelisms import ParallelDims
from torchtitan.utils import common_utils as utils


def loss_fn(pred, labels):
    return cross_entropy(pred.flatten(0, 1), labels.flatten(0, 1))


def validate(
    model: Any,  # otherwise would have to update union type for every architecture
    data_loader: DPAwareDataLoader,
    logger: Logger,
    metric_logger: MetricLogger,
    parallel_dims: ParallelDims,
    gc_handler: utils.GarbageCollection,
    gpu_memory_monitor: GPUMemoryMonitor,
    color: Union[utils.Color, utils.NoColor],
    train_step: int,  # for aim tracking of evaluation to be tracked correctly
    num_flop_per_token: int,
    gpu_peak_flops: int,
    dp_rank: int,
    dp_mesh,
    world_size: int,
    enable_compiled_autograd: bool,
    device: str
):
    model.eval()
    time_last_log = (
        time.perf_counter()
    )  # we are swiching to validation so we don't calculate time in transition
    gpu_memory_monitor.reset_peak_stats()
    eval_state = TrainState()
    total_n_tokens = 0
    total_loss = 0
    total_perplexity = 0

    total_eval_time = 0
    total_data_loading_time = 0

    val_data_iterator = iter(data_loader)

    train_context = utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        enable_compiled_autograd,
    )

    eval_state.step = 0
    while True:
        eval_state.step += 1
        gc_handler.run(eval_state.step)

        # get batch
        data_load_start = time.perf_counter()
        logger.debug("validation step")

        batch = next(val_data_iterator, None)
        # create a binary array representing if the ith rank has exhausted its data
        dataloader_finished_status = torch.zeros(world_size).to(device)
        if not batch:
            dataloader_finished_status[dp_rank] = 1

        dist.all_reduce(dataloader_finished_status)
        if dataloader_finished_status.sum() > 0:
            break

        input_ids, labels = batch

        n_tokens_in_curr = labels.numel()
        input_ids = input_ids.cuda()
        labels = labels.cuda()

        with train_context(), torch.no_grad():
            logger.debug("enter context")
            total_n_tokens += n_tokens_in_curr  # we only add to the total tokens if we actually run a prediction
            total_data_loading_time += time.perf_counter() - data_load_start
            pred = model(input_ids)
            loss = loss_fn(pred, labels)
            del pred

        time_delta = time.perf_counter() - time_last_log
        total_eval_time += time_delta
        total_loss += loss
        total_perplexity += 2**loss
        wps = n_tokens_in_curr / (time_delta * parallel_dims.model_parallel_size)
        mfu = 100 * num_flop_per_token * wps / gpu_peak_flops
        gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

        logger.info(
            "context: valid  "
            f"{color.cyan}step: {eval_state.step}  "
            f"{color.green}loss: {loss:7.4f}  "
            f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
            f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
            f"{color.blue}wps: {round(wps):,}  "
            f"{color.magenta}mfu: {mfu:.2f}%{color.reset}"
        )
        time_last_log = time.perf_counter()

    avg_time_end_to_end = total_eval_time / eval_state.step
    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()

    global_total_loss = utils.dist_mean(total_loss, dp_mesh)
    global_total_perplexity = utils.dist_mean(total_perplexity, dp_mesh)
    
    metrics = {
        "valid/loss_metrics/global_avg_loss": global_total_loss / eval_state.step,
        "valid/loss_metrics/global_avg_perplexity": global_total_perplexity / eval_state.step,
        "valid/wps": total_n_tokens / total_eval_time,
        "valid/mfu(%)": 100
        * num_flop_per_token
        * (total_n_tokens / total_eval_time)
        / gpu_peak_flops,
        "valid/time_metrics/end_to_end(s)": avg_time_end_to_end,
        "valid/time_metrics/data_loading(s)": total_data_loading_time,
        "valid/time_metrics/data_loading(%)": total_data_loading_time / total_eval_time,
        "valid/memory/max_active(GiB)": gpu_mem_stats.max_active_gib,
        "valid/memory/max_active(%)": gpu_mem_stats.max_active_pct,
        "valid/memory/max_reserved(GiB)": gpu_mem_stats.max_reserved_gib,
        "valid/memory/max_reserved(%)": gpu_mem_stats.max_reserved_pct,
        "valid/memory/num_alloc_retries": gpu_mem_stats.num_alloc_retries,
        "valid/memory/num_ooms": gpu_mem_stats.num_ooms,
    }
    metric_logger.log(metrics, step=train_step)
    gpu_memory_monitor.reset_peak_stats()

    if loss:
        del loss

    del val_data_iterator, data_loader

    gc_handler.run(gc_handler.gc_freq) # gc at the end
    model.train()
