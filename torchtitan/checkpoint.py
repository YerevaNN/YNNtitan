# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import enum
import functools
import os
import re
import shutil
import time
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from multiprocessing import get_context
from typing import Any, Dict, List, Union, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import DataLoader
from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.logging import init_logger, logger
from torch.distributed.fsdp import StateDictType
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp._common_utils import FSDP_WRAPPED_MODULE
from torch.nn.parallel import DistributedDataParallel as DDP


class IntervalType(enum.Enum):
    SECONDS = enum.auto()
    STEPS = enum.auto()


class AsyncMode(str, enum.Enum):
    DISABLED = "disabled"
    ASYNC = "async"
    ASYNC_WITH_PINNED_MEM = "async_with_pinned_mem"


@dataclass
class TrainState(Stateful):
    step: int = 0
    global_avg_losses: List[float] = field(default_factory=list)
    global_max_losses: List[float] = field(default_factory=list)
    global_avg_perplexities: List[float] = field(default_factory=list)
    global_max_perplexities: List[float] = field(default_factory=list)
    log_steps: List[int] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        # Only checkpoint global_avg_losses and global_max_losses per log frequency
        # to avoid sync overhead in every iteration.
        global_avg_losses_bytes = BytesIO()
        torch.save(self.global_avg_losses, global_avg_losses_bytes)
        global_max_losses_bytes = BytesIO()
        torch.save(self.global_max_losses, global_max_losses_bytes)
        log_steps_bytes = BytesIO()
        torch.save(self.log_steps, log_steps_bytes)
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "global_avg_losses": global_avg_losses_bytes,
            "global_max_losses": global_max_losses_bytes,
            "log_steps": log_steps_bytes,
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        state_dict["global_avg_losses"].seek(0)
        self.global_avg_losses = torch.load(
            state_dict["global_avg_losses"], weights_only=False
        )
        state_dict["global_max_losses"].seek(0)
        self.global_max_losses = torch.load(
            state_dict["global_max_losses"], weights_only=False
        )
        state_dict["log_steps"].seek(0)
        self.log_steps = torch.load(state_dict["log_steps"], weights_only=False)


class ModelWrapper(Stateful):
    def __init__(self, model: Union[nn.Module, List[nn.Module]], vocab_size: Optional[int] = None) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.target_vocab_size = vocab_size
        self.original_embedding_weights = {}

    def state_dict(self) -> None:
        # Flatten state dicts and drop non-critical, recomputable buffers such as 'freqs_cis'
        merged = {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}
        filtered = {k: v for k, v in merged.items() if not k.endswith("freqs_cis")}
        return filtered

    def _resize_token_embeddings(self, new_vocab_size: int):
        """Resize token embeddings to match checkpoint vocabulary size"""
        for model in self.model:
            if hasattr(model, 'tok_embeddings'):
                old_embeddings = model.tok_embeddings
                old_vocab_size = old_embeddings.weight.size(0)
                
                if old_vocab_size != new_vocab_size:
                    logger.info(f"Resizing token embeddings from {old_vocab_size} to {new_vocab_size}")
                    
                    # Save original weights if we're expanding for later restoration
                    if new_vocab_size > old_vocab_size:
                        self.original_embedding_weights[id(model)] = old_embeddings.weight.data.clone()
                    
                    # Create new embedding layer with the target size
                    new_embeddings = nn.Embedding(new_vocab_size, old_embeddings.weight.size(1))
                    new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
                    
                    # Copy existing weights
                    min_vocab_size = min(old_vocab_size, new_vocab_size)
                    new_embeddings.weight.data[:min_vocab_size] = old_embeddings.weight.data[:min_vocab_size]
                    
                    # Initialize new tokens if expanding
                    if new_vocab_size > old_vocab_size:
                        # Initialize new token embeddings with Qwen3 embedding init
                        # Match Qwen3Transformer.init_weights embed_std
                        embed_std = 0.006
                        nn.init.trunc_normal_(
                            new_embeddings.weight.data[old_vocab_size:],
                            mean=0.0,
                            std=embed_std,
                        )
                    
                    # Replace the embedding layer
                    model.tok_embeddings = new_embeddings
                    model.vocab_size = new_vocab_size
                    if hasattr(model.model_args, 'vocab_size'):
                        model.model_args.vocab_size = new_vocab_size

                    # If model has a separate output head, resize it as well
                    if hasattr(model, 'output') and isinstance(model.output, nn.Linear) and model.output is not None:
                        old_output = model.output
                        old_out_vocab = old_output.weight.size(0)
                        if old_out_vocab != new_vocab_size:
                            logger.info(f"Resizing output head from {old_out_vocab} to {new_vocab_size}")
                            # Create new output with same input dim and no bias
                            new_output = nn.Linear(old_output.in_features, new_vocab_size, bias=False)
                            new_output.to(old_output.weight.device, dtype=old_output.weight.dtype)
                            # Copy overlapping rows
                            min_out = min(old_out_vocab, new_vocab_size)
                            new_output.weight.data[:min_out] = old_output.weight.data[:min_out]
                            # Initialize any new rows to match Qwen3 final layer init
                            if new_vocab_size > old_out_vocab:
                                final_out_std = (model.model_args.dim ** -0.5) if hasattr(model, 'model_args') and hasattr(model.model_args, 'dim') else 0.02
                                cutoff_factor = 3
                                # Use trunc_normal_ limited to +/- 3 std like model init
                                nn.init.trunc_normal_(
                                    new_output.weight.data[old_out_vocab:],
                                    mean=0.0,
                                    std=final_out_std,
                                    a=-cutoff_factor * final_out_std,
                                    b=cutoff_factor * final_out_std,
                                )
                            # Replace output head
                            model.output = new_output

    def _get_vocab_size_from_checkpoint_info(self, checkpoint_path: str) -> Optional[int]:
        """Try to extract vocabulary size from checkpoint metadata"""
        try:
            metadata_path = os.path.join(checkpoint_path, ".metadata")
            if os.path.exists(metadata_path):
                # Try to load metadata to get shape info if available
                # This is a simple heuristic - in practice you might need to check actual tensor files
                pass
        except:
            pass
        return None

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Check for vocabulary size mismatch in token embeddings
        vocab_mismatch_detected = False
        checkpoint_vocab_size = None
        
        for key, value in state_dict.items():
            if 'tok_embeddings.weight' in key:
                checkpoint_vocab_size = value.shape[0]
                # Check if current model has different vocab size
                for model in self.model:
                    if hasattr(model, 'tok_embeddings'):
                        current_vocab_size = model.tok_embeddings.weight.size(0)
                        if current_vocab_size != checkpoint_vocab_size:
                            vocab_mismatch_detected = True
                            logger.info(f"Detected vocabulary size mismatch: checkpoint={checkpoint_vocab_size}, current={current_vocab_size}")
                            break
                break
        
        if vocab_mismatch_detected and checkpoint_vocab_size:
            # Temporarily resize to match checkpoint
            logger.info(f"Temporarily resizing embeddings to match checkpoint size: {checkpoint_vocab_size}")
            for model in self.model:
                if hasattr(model, 'tok_embeddings'):
                    current_vocab_size = model.tok_embeddings.weight.size(0)
                    if current_vocab_size != checkpoint_vocab_size:
                        self._resize_token_embeddings(checkpoint_vocab_size)
            
            # Load the state dict
            func = functools.partial(
                set_model_state_dict,
                model_state_dict=state_dict,
                options=StateDictOptions(strict=False),
            )
            list(map(func, self.model))
            
            # Resize back to target vocabulary size if needed
            if self.target_vocab_size and self.target_vocab_size != checkpoint_vocab_size:
                logger.info(f"Resizing embeddings back to target size: {self.target_vocab_size}")
                for model in self.model:
                    if hasattr(model, 'tok_embeddings'):
                        self._resize_token_embeddings(self.target_vocab_size)
        else:
            # Normal loading without vocabulary mismatch
            func = functools.partial(
                set_model_state_dict,
                model_state_dict=state_dict,
                options=StateDictOptions(strict=False),
            )
            list(map(func, self.model))


class OptimizerWrapper(Stateful):
    def __init__(
        self,
        model: Union[nn.Module, List[nn.Module]],
        optim: Union[torch.optim.Optimizer, List[torch.optim.Optimizer]],
    ) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.optim = [optim] if isinstance(optim, torch.optim.Optimizer) else optim

    def state_dict(self) -> None:
        func = functools.partial(
            get_optimizer_state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        return {k: v for sd in map(func, self.model, self.optim) for k, v in sd.items()}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        func = functools.partial(
            set_optimizer_state_dict,
            optim_state_dict=state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        list(map(func, self.model, self.optim))


class Terminate:
    pass


class SaveDone:
    pass


def checkpoint_mp(recv, send, log_level):
    init_logger(log_level)
    os.environ["MASTER_PORT"] = str(int(os.environ["MASTER_PORT"]) + 2)
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group()
    try:
        while True:
            logger.debug("Checkpoint background process is done.")
            send.put(SaveDone())
            logger.debug("Wait for the new state_dict.")
            obj = recv.get()
            logger.debug("Received the new state_dict.")
            if isinstance(obj, Terminate):
                logger.info("Terminating the checkpoint background process.")
                return
            assert isinstance(obj, tuple)
            begin = time.monotonic()
            state, checkpoint_id = obj
            dcp.save(state, checkpoint_id=checkpoint_id)
            logger.info(
                "Finish saving the checkpoint in the background process in "
                f"{time.monotonic() - begin:.2f} seconds."
            )
    finally:
        logger.info("Destroying the process group.")
        dist.destroy_process_group()


class CheckpointManager:
    def __init__(
        self,
        dataloader: DataLoader,
        model_parts: List[nn.Module],
        optimizers: List[torch.optim.Optimizer],
        lr_schedulers: List[torch.optim.lr_scheduler.LRScheduler],
        states: Dict[str, Any],
        job_config: JobConfig,
        experiment_hash: str,
    ) -> None:
        ckpt_config = job_config.checkpoint
        self.enable_checkpoint = ckpt_config.enable_checkpoint
        self.keep_latest_k = ckpt_config.keep_latest_k

        if not self.enable_checkpoint:
            return
        """
        Note: Pipeline Parallelism and Virtual Stages

        1. even for simple PP schedules, there is a separate optimizer each PP rank.
        rank0's optimizer would have a param_group[0] which refers to layers.0 in the original model.
        rank1's would _also_ have a param_group[0], since it's index based, but referring to layers.1.
        When saving, these collide and one of them is lost.  Then when reloading, only one stage can
        restore its optimizer states, others will error.

            The solution to this problem is optimizer flattening: it landed in #127071 and is enabled in TorchTitan
            by passing the 'flatten_optimizer_state_dict' kwarg to DCP functions called in the OptimizerWrapper.

        2. With complex PP schedules, we have multiple model chunks per pp rank. This compounds challenge (1) by also
        requiring us to reason about multiple 'optim' objects locally.

            We solve this in the Model and Optimizer wrapper classes by flattening the state dicts from each object
            into one state dict before saving/loading. We rely on the individual state_dicts to not collide,
            which is gauranteed for the model by correct pipeline splitting and for the optimizer by the flattening
            support described in (1).

        3. LR schedulers also index model states like optimizers and would need to be flattened properly to support
        resharding.  Unfortunately, the implementations of different lr_schedulers do not follow a clear pattern like
        optimizers do, so it's hard to write a generic 'flattener' utility.

            TODO: This is currently unsolved and needs a fix.
        """
        assert len(model_parts) == len(
            optimizers
        ), "Must pass one optimizer per model part"
        assert len(model_parts) == len(
            lr_schedulers
        ), "Must pass one lr_scheduler per model part"

        assert len(model_parts) == len(
            optimizers
        ), "Must pass one optimizer per model part"
        assert len(model_parts) == len(
            lr_schedulers
        ), "Must pass one lr_scheduler per model part"

        self.states = states

        # Extract target vocabulary size from model if available
        target_vocab_size = None
        for model in model_parts:
            if hasattr(model, 'vocab_size'):
                target_vocab_size = model.vocab_size
                break
            elif hasattr(model, 'model_args') and hasattr(model.model_args, 'vocab_size'):
                target_vocab_size = model.model_args.vocab_size
                break

        self.states.update(
            {
                "model": ModelWrapper(model_parts, vocab_size=target_vocab_size),
                "optimizer": OptimizerWrapper(model_parts, optimizers),
                "dataloader": dataloader,
            }
        )
        if len(lr_schedulers) == 1:
            self.states["lr_scheduler"] = lr_schedulers[0]
        else:
            # For now, pipeline-parallel with looped schedules does not support resharding for lr_scheduler.
            # It should only support saving and loading a distributed checkpoint with the same number of pp ranks
            for idx, lr_scheduler in enumerate(lr_schedulers):
                self.states[f"lr_scheduler_{idx}"] = lr_scheduler

        if (
            job_config.model_download_export.to_hf
            or job_config.model_download_export.to_titan
        ):
            self.save_folder = os.path.join(
                job_config.job.dump_folder, ckpt_config.save_folder
            )
        else:
            self.save_folder = os.path.join(
                job_config.job.dump_folder,
                os.path.join(ckpt_config.save_folder, experiment_hash),
            )
        self.load_folder = os.path.join(
            job_config.job.dump_folder, ckpt_config.load_folder
        )
        self.interval_type = (
            IntervalType.SECONDS
            if ckpt_config.interval_type == "seconds"
            else IntervalType.STEPS
        )
        self.interval = ckpt_config.interval
        self.begin_time = 0
        self.time_sync_work = None
        self.time_sync_result = None
        self.pg = dist.new_group(backend="gloo")

        self.model_weights_only = ckpt_config.model_weights_only
        self.export_dtype = TORCH_DTYPE_MAP[ckpt_config.export_dtype]

        self.mp = None
        async_mode = ckpt_config.async_mode.lower()
        if async_mode == AsyncMode.DISABLED:
            self.async_mode = AsyncMode.DISABLED
        elif async_mode == AsyncMode.ASYNC:
            self.async_mode = AsyncMode.ASYNC
            self.async_future = None
        elif async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.async_mode = AsyncMode.ASYNC_WITH_PINNED_MEM
            ctx = get_context("spawn")
            self.mp_queue_send = ctx.Queue()
            self.mp_queue_recv = ctx.Queue()
            self.mp = ctx.Process(
                target=checkpoint_mp,
                args=(
                    self.mp_queue_send,
                    self.mp_queue_recv,
                    job_config.logging.log_level,
                ),
                daemon=True,
            )
            self.mp.start()
            self.cpu_offload_state_dict = None
            self.staging = False
            self.staging_state_dict = None
            self.staging_id = None
            self.staging_stream = torch.cuda.Stream()
        else:
            raise ValueError(f"Unkown checkpoint async_mode {ckpt_config.async_mode}")

        logger.info(
            f"Checkpointing active. Checkpoints will be loaded from {self.load_folder} and saved to {self.save_folder}"
        )

    def __del__(self):
        if self.enable_checkpoint and self.mp and self.mp.is_alive():
            self.mp_queue_send.put(Terminate())
            self.mp.join()

    def reset(self) -> None:
        self.begin_time = time.monotonic()

    def _create_checkpoint_id(self, step: int, folder: str) -> str:
        return os.path.join(folder, f"step-{step}")

    def _save_last_step(self, curr_step: int) -> None:
        # We only consider saving weights only at the end of the training. So
        # this won't affect preemption and training resume. We also only allow
        # dtype conversion when we are checkpoint model weights only and the
        # current dtype is not the same as the export dtype at the end of the training.
        if self.model_weights_only:
            # We update self.states to keep the model only.
            # After this update, self.states = {
            #      'tok_embeddings.weight':...,
            #      'layers.0.attention.wq.weight': ...
            # }.
            self.states = self.states["model"].state_dict()

            # For now, we will manually pop the freqs_cis buffer, as we made this permanent
            # temporarily and we don't want to include it in the exported state_dict.
            # Context: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama/model.py#L348
            self.states.pop("freqs_cis")

            if self.export_dtype != torch.float32:
                self.states = {
                    k: v.to(self.export_dtype) for k, v in self.states.items()
                }
            logger.info(
                f"Saving a model weights only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")

        dcp.save(
            self.states,
            checkpoint_id=self._create_checkpoint_id(curr_step, self.save_folder),
        )
        self.reset()

    def _should_save(self, curr_step: int, force: bool = False) -> bool:
        if not self.enable_checkpoint:
            return False

        if not force:
            if self.interval_type == IntervalType.STEPS and not (
                curr_step % self.interval == 0
            ):
                return False
            if self.interval_type == IntervalType.SECONDS:
                time_sync_result = (time.monotonic() - self.begin_time) >= self.interval
                self.time_sync_result = torch.tensor(int(time_sync_result))
                if self.time_sync_work is None:
                    self.time_sync_work = dist.all_reduce(
                        self.time_sync_result, group=self.pg, async_op=True
                    )
                    return False
                elif curr_step % 5 == 4:
                    self.time_sync_work.wait()
                    self.time_sync_work = None
                    time_sync_result = self.time_sync_result.item()
                    self.time_sync_result = None
                    if time_sync_result == 0:
                        return False
                else:
                    return False

        if self.time_sync_work:
            self.time_sync_work.wait()
            self.time_sync_work = None
            self.time_sync_result = None

        return True

    def _async_wait(self) -> None:
        if self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            logger.debug(
                f"Waiting for the background process to finish, {time.monotonic()=}.:.2f"
            )
            if not self.mp.is_alive():
                raise RuntimeError("The checkpoint background process is dead.")
            _ = self.mp_queue_recv.get()
        elif self.async_mode == AsyncMode.ASYNC:
            if self.async_future is not None:
                self.async_future.result()

    def _async_with_pinned_memory(self, checkpoint_id: str) -> None:
        try:
            from torch.distributed._state_dict_utils import (
                _copy_state_dict,
                _create_cpu_state_dict,
            )
        except ImportError as e:
            raise ImportError(
                "Please install the latest PyTorch nightly to use async checkpointing with pinned memory."
            ) from e
        state_dict = dcp.state_dict_saver._stateful_to_state_dict(self.states)
        if self.cpu_offload_state_dict is None:
            logger.debug(f"Preparing the CPU memory, {time.monotonic()=}.:.2f")
            self.cpu_offload_state_dict = _create_cpu_state_dict(
                state_dict, pin_memory=True
            )

        logger.debug(f"Staging the state_dict, {time.monotonic()=}.:.2f")
        with torch.cuda.stream(self.staging_stream):
            self.cpu_offload_state_dict = _copy_state_dict(
                state_dict,
                self.cpu_offload_state_dict,
                non_blocking=True,
            )
            self.staging = True
            self.staging_state_dict = state_dict
            self.staging_id = checkpoint_id

    def save(self, curr_step: int, force: bool = False) -> None:
        """
        force = True will force the checkpoint to be saved, even if the interval
        has not been reached.
        This only happens when train_state.step == job_config.training.steps, or
        for initial seed checkpoint.
        """
        if not self._should_save(curr_step, force):
            return

        begin = time.monotonic()
        checkpoint_id = self._create_checkpoint_id(curr_step, self.save_folder)
        self._async_wait()
        if force:
            self._save_last_step(curr_step)
        elif self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM:
            self._async_with_pinned_memory(checkpoint_id)
        elif self.async_mode == AsyncMode.ASYNC:
            self.async_future = dcp.async_save(
                self.states, checkpoint_id=checkpoint_id, process_group=self.pg
            )
        else:
            dcp.save(self.states, checkpoint_id=checkpoint_id)
        self.reset()
        self._purge_stale_checkpoints()

        logger.info(
            "Finished saving the checkpoint (or staging if async is enabled)"
            f"in {time.monotonic() - begin:.2f} seconds."
        )

    def maybe_wait_for_staging(self) -> None:
        if (
            self.enable_checkpoint
            and self.async_mode == AsyncMode.ASYNC_WITH_PINNED_MEM
            and self.staging
        ):
            logger.debug(f"Waiting for staging, {time.monotonic()=:.2f}.")
            self.staging_stream.synchronize()
            logger.debug(
                f"Sending the state dict to the background process, {time.monotonic()=:.2f}."
            )
            self.mp_queue_send.put((self.staging_state_dict, self.staging_id))
            self.staging = False

    def load(self, step: int = -1) -> bool:
        if not self.enable_checkpoint:
            return False
        if not os.path.isdir(self.load_folder):
            return False
        if step != -1 and not os.path.isdir(
            self._create_checkpoint_id(step, self.load_folder)
        ):
            return False

        if step == -1:
            step_counts = []
            for filename in os.listdir(self.load_folder):
                match = re.search(r"step-(\d+)", filename)
                metadata_probe = os.path.join(self.load_folder, filename, ".metadata")
                if match and os.path.isfile(metadata_probe):
                    step_counts.append(int(match.group(1)))
            if not step_counts:
                return False
            step = max(step_counts)

        # We won't have optimizer states to load, if we are loading a seed checkpoint
        states = {"model": self.states["model"]} if step == 0 else self.states
        # PyTorch bug: (pytorch/pytorch#138575)
        # dcp.load() replaces the values of stateful elements in `states` with new objects
        # from loading the checkpoint, in addition to updating the states of the original
        # objects from `states` in-place. This is a problem because the state_dict no longer
        # refers to the objects being used in the train loop, meaning any future checkpoints
        # will not include updates to these objects (such as updated optimizer states, etc.)
        original_stateful_states = {
            k: v for k, v in states.items() if isinstance(v, Stateful)
        }
        logger.info(f"Loading the checkpoint at step {step}.")
        begin = time.monotonic()
        dcp.load(
            original_stateful_states,
            checkpoint_id=self._create_checkpoint_id(step, self.load_folder),
        )
        logger.info(
            f"Finished loading the checkpoint in {time.monotonic() - begin:.2f} seconds."
        )
        # bugfix from above: restore the original stateful objects,
        # whose states were already updated in-place by dcp.load()
        # for k, v in original_stateful_states.items():
        #     states[k].load_state_dict(v)
        return True

    def _purge_stale_checkpoints(self):
        if self.keep_latest_k > 0:
            discovered_checkpoints = []
            for filename in os.listdir(self.save_folder):
                match = re.search(r"step-(\d+)", filename)
                path = os.path.join(self.save_folder, filename)
                discovered_checkpoints.append((int(match.group(1)), path))

            discovered_checkpoints.sort()
            to_delete = discovered_checkpoints[: -1 * self.keep_latest_k]

            for _, path in to_delete:
                logger.info(f"Deleting old checkpoint {path}")
                shutil.rmtree(path, ignore_errors=True)
