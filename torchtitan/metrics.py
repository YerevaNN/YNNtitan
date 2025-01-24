# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import namedtuple
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from torchtitan.aim import AimLogger
from torchtitan.config_manager import JobConfig
from torchtitan.logging import logger
from torchtitan.parallelisms import ParallelDims

# named tuple for passing GPU memory stats for logging
GPUMemStats = namedtuple(
    "GPUMemStats",
    [
        "max_active_gib",
        "max_active_pct",
        "max_reserved_gib",
        "max_reserved_pct",
        "num_alloc_retries",
        "num_ooms",
    ],
)


class GPUMemoryMonitor:
    def __init__(self, device: str = "cuda:0"):
        self.device = torch.device(device)  # device object
        self.device_name = torch.cuda.get_device_name(self.device)
        self.device_index = torch.cuda.current_device()
        self.device_capacity = torch.cuda.get_device_properties(
            self.device
        ).total_memory
        self.device_capacity_gib = self._to_gib(self.device_capacity)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    def _to_gib(self, memory_in_bytes):
        # NOTE: GiB (gibibyte) is 1024, vs GB is 1000
        _gib_in_bytes = 1024 * 1024 * 1024
        memory_in_gib = memory_in_bytes / _gib_in_bytes
        return memory_in_gib

    def _to_pct(self, memory):
        return 100 * memory / self.device_capacity

    def get_peak_stats(self):
        cuda_info = torch.cuda.memory_stats(self.device)

        max_active = cuda_info["active_bytes.all.peak"]
        max_active_gib = self._to_gib(max_active)
        max_active_pct = self._to_pct(max_active)

        max_reserved = cuda_info["reserved_bytes.all.peak"]
        max_reserved_gib = self._to_gib(max_reserved)
        max_reserved_pct = self._to_pct(max_reserved)

        num_retries = cuda_info["num_alloc_retries"]
        num_ooms = cuda_info["num_ooms"]

        if num_retries > 0:
            logger.warning(f"{num_retries} CUDA memory allocation retries.")
        if num_ooms > 0:
            logger.warning(f"{num_ooms} CUDA OOM errors thrown.")

        return GPUMemStats(
            max_active_gib,
            max_active_pct,
            max_reserved_gib,
            max_reserved_pct,
            num_retries,
            num_ooms,
        )

    def reset_peak_stats(self):
        torch.cuda.reset_peak_memory_stats()


def build_gpu_memory_monitor():
    gpu_memory_monitor = GPUMemoryMonitor("cuda")
    logger.info(
        f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
        f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory"
    )

    return gpu_memory_monitor


class MetricLogger:
    def __init__(self, hash, experiment_name, log_dir, save_aim_folder, enable_aim):
        self.writer: Optional[AimLogger] = None
        if enable_aim:
            if hash is not None:
                self.writer = AimLogger(save_aim_folder, run_hash=hash)
            elif experiment_name is not None:
                self.writer = AimLogger(save_aim_folder, experiment=experiment_name)
            else:
                self.writer = AimLogger(save_aim_folder)
            self.experiment_hash = self.writer.experiment.hash
        else:
            self.experiment_hash = "default"

    def log(self, metrics: Dict[str, Any], step: int):
        if self.writer is not None:
            self.writer.log_metrics(metrics, step)

    def close(self):
        if self.writer is not None:
            self.writer.finalize()

    def log_hparams(self, config):
        if self.writer is not None:
            self.writer.experiment["hparams"] = config


def build_metric_logger(job_config: JobConfig, parallel_dims: ParallelDims):
    """
    parallel_dims is used to determine the rank to log metrics from if 'aim_config.rank_0_only=True'.
    In that case, `_get_metrics_rank` will be used to calculate which rank acts as 'rank 0'. This is
    intended to allow logging from the 0th rank within the last pipeline stage group, in case pipeline
    parallelism is enabled, without forcing logging from all ranks to capture loss information.
    """
    dump_dir = job_config.job.dump_folder
    aim_config = job_config.metrics
    save_aim_folder = os.path.join(
        job_config.job.dump_folder, aim_config.save_aim_folder
    )
    # since we don't have run id, use current minute as the identifier
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(dump_dir, datetime_str)

    enable_aim = aim_config.enable_aim
    if enable_aim:
        logger.info(
            f"Metrics logging active. Aim logs will be saved at /{save_aim_folder}"
        )
        enable_aim = torch.distributed.get_rank() == 0

    metric_logger = MetricLogger(
        job_config.metrics.aim_hash,
        job_config.metrics.aim_experiment_name,
        log_dir,
        save_aim_folder,
        enable_aim,
    )

    experiment_hash_list = [metric_logger.experiment_hash]
    # broadcast aim experiment hash to all ranks
    torch.distributed.broadcast_object_list(experiment_hash_list, src=0)
    metric_logger.experiment_hash = experiment_hash_list[0]
    return metric_logger
