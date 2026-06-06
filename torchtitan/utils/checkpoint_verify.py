# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.checkpoint import TrainState
from torchtitan.logging import logger
from torchtitan.utils import common_utils as utils


def _reference_loss_at_step(train_state: TrainState, step: int) -> Optional[float]:
    if step not in train_state.log_steps:
        return None
    idx = train_state.log_steps.index(step)
    if idx >= len(train_state.global_avg_losses):
        return None
    return train_state.global_avg_losses[idx]


@torch.no_grad()
def verify_checkpoint_after_load(
    model_parts: List[nn.Module],
    data_iterator: Iterable,
    loss_fn: Callable,
    train_context,
    train_state: TrainState,
    num_microbatches: int,
    parallel_dims,
    dp_mesh: Optional[DeviceMesh],
    loaded_step: int,
    max_delta: float,
) -> Tuple[float, Optional[float]]:
    """
    Forward-only loss on the next training microbatches. The caller must restore
    dataloader state afterward so the train loop sees the same batches.
    Returns (measured_global_avg_loss, reference_loss_from_train_state or None).
    """
    for mod in model_parts:
        mod.train()

    local_losses: List[float] = []
    for _ in range(num_microbatches):
        batch = next(data_iterator)
        input_ids, labels = batch
        input_ids = input_ids.cuda()
        labels = labels.cuda()
        with train_context():
            pred = model_parts[0](input_ids)
            cur_loss = loss_fn(pred, labels)
        local_losses.append(cur_loss.item())
        del pred, input_ids, labels

    avg_loss = sum(local_losses) / len(local_losses)
    if parallel_dims.dp_enabled:
        global_avg_loss = utils.dist_mean(avg_loss, dp_mesh)
    else:
        global_avg_loss = avg_loss

    next_step = loaded_step + 1
    ref_loss = _reference_loss_at_step(train_state, next_step)
    ref_step = next_step
    if ref_loss is None:
        ref_loss = _reference_loss_at_step(train_state, loaded_step)
        ref_step = loaded_step
    if ref_loss is not None:
        delta = abs(global_avg_loss - ref_loss)
        if delta <= max_delta:
            logger.info(
                "Checkpoint verify: forward loss %.4f at resume step %s "
                "(checkpoint logged %.4f at step %s, delta %.4f <= %.4f).",
                global_avg_loss,
                next_step,
                ref_loss,
                ref_step,
                delta,
                max_delta,
            )
        else:
            logger.warning(
                "Checkpoint verify: forward loss %.4f at resume step %s "
                "differs from checkpoint logged %.4f at step %s by %.4f (threshold %.4f). "
                "Weights or data stream may not match the saved run.",
                global_avg_loss,
                next_step,
                ref_loss,
                ref_step,
                delta,
                max_delta,
            )
    else:
        logger.info(
            "Checkpoint verify: forward loss %.4f at resume step %s "
            "(no logged loss for steps %s or %s in train_state).",
            global_avg_loss,
            next_step,
            next_step,
            loaded_step,
        )

    return global_avg_loss, ref_loss
