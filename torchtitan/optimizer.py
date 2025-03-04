# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from enum import Enum

import torch
from torch.optim.lr_scheduler import LambdaLR
from torchtitan.config_manager import JobConfig


def build_optimizers(model_parts, job_config: JobConfig):
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """

    def _build_optimizer(model):
        name = job_config.optimizer.name
        lr = job_config.optimizer.lr
        fused = job_config.optimizer.fused

        # Common parameters for both optimizers
        optimizer_kwargs = {
            "lr": lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": fused,
            "foreach": not fused,
        }
        if name == "Adam":
            # TODO: make the optimizer options configurable by toml/cmd args
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

        return optimizer

    class OptimizersContainer:
        """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

        def __init__(self, optimizers):
            self.optimizers = optimizers

        def step(self):
            for optimizer in self.optimizers:
                optimizer.step()

        def zero_grad(self):
            for optimizer in self.optimizers:
                optimizer.zero_grad()

    return OptimizersContainer([_build_optimizer(model) for model in model_parts])


def linear_warmup(warmup_steps: int, current_step: int) -> float:
    """Computes the linear warmup scaling factor."""
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive.")
    return float((current_step + 1) / (warmup_steps + 1))


# Decay functions
def linear_decay(decay_steps: int, current_step: int, start_step: int) -> float:
    """Computes the linear decay scaling factor."""
    if decay_steps <= 0:
        raise ValueError("decay_steps must be positive.")
    progress = float((current_step - start_step) / decay_steps)
    return max(0.0, 1 - progress)


def cosine_decay(decay_steps: int, current_step: int, start_step: int) -> float:
    """Computes the cosine decay scaling factor."""
    if decay_steps <= 0:
        raise ValueError("decay_steps must be positive.")
    current_step = min(current_step - start_step, decay_steps)
    return 0.5 * (1 + math.cos(math.pi * current_step / decay_steps))


class Decay(Enum):
    LINEAR = functools.partial(linear_decay)
    COSINE = functools.partial(cosine_decay)

    @staticmethod
    def from_string(decay_type: str) -> "Decay":
        """Converts a string to the corresponding Decay enum value."""
        try:
            return Decay[decay_type.upper()]
        except KeyError as e:
            raise ValueError(
                f"Invalid decay type: {decay_type}. Expected one of {list(Decay.__members__.keys())}"
            ) from e


def warmup_stable_decay(
    decay_type: Decay,
    warmup_steps: int,
    decay_steps: int,
    training_steps: int,
    current_step: int,
) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    start_decay_step = training_steps - decay_steps

    if current_step < warmup_steps:
        # warmup phase
        curr_adjustment = linear_warmup(warmup_steps, current_step)
        return linear_warmup(warmup_steps, current_step)

    elif (current_step >= warmup_steps) and (current_step < start_decay_step):
        # stable phase, no adjustment to lr
        return 1.0

    else:
        # decay phase supporting multiple decay functions
        return decay_type.value(decay_steps, current_step, start_decay_step)


# implementation of WSD-S scheduler
def warmup_stable_decay_simplified(
    decay_type: Decay,
    warmup_steps: int,
    decay_steps_perc: float,
    num_decays: int,
    training_steps: int,
    current_step: int,
) -> float:
    # num steps for each decay
    per_decay_num_steps = training_steps // num_decays
    # current decay index
    decay_index = math.ceil(current_step / per_decay_num_steps)
    # the step at which lr is decayed
    decay_at_step = decay_index * per_decay_num_steps
    # number of decay steps
    if decay_index == 1:
        # make sure the decay_steps_perc does not include the warmup_steps
        decay_steps_perc = min(decay_steps_perc, 1 - warmup_steps / decay_at_step)

    decay_steps = int(decay_at_step * decay_steps_perc)
    # the step at which to start the decay
    start_decay_step = decay_at_step - decay_steps

    if current_step < warmup_steps:
        # warmup phase
        curr_adjustment = current_step / warmup_steps
    elif current_step < start_decay_step:
        # stable phase, no adjustment to lr
        curr_adjustment = 1.0
    else:
        # decay phase supporting multiple decay functions
        curr_adjustment = decay_type.value(decay_steps, current_step, start_decay_step)

    return curr_adjustment


def build_lr_schedulers(optimizers, job_config: JobConfig) -> LambdaLR:
    def _build_lr_scheduler(optimizer):
        """Build a linear warmup optionally stable and linear decay scheduler"""
        warmup_steps = int(job_config.training.warmup_steps)
        post_warmup_steps = float(max(1, job_config.training.steps - warmup_steps))

        decay_steps_perc = job_config.training.decay_steps_perc
        num_decays = job_config.training.num_decays
        decay_type = Decay.from_string(job_config.training.decay_type)

        # lr_lambda = functools.partial(
        #     warmup_stable_decay, decay_type, warmup_steps, decay_steps, job_config.training.steps
        # )
        lr_lambda = functools.partial(
            warmup_stable_decay_simplified,
            decay_type,
            warmup_steps,
            decay_steps_perc,
            num_decays,
            job_config.training.steps,
        )
        warmup_stable_decay_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return warmup_stable_decay_scheduler

    class SchedulersContainer:
        """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

        def __init__(self, schedulers):
            self.schedulers = schedulers

        def step(self):
            for schedulers in self.schedulers:
                schedulers.step()

        @property
        def last_lr(self):
            return self.schedulers[0].get_last_lr()[0]

    return SchedulersContainer(
        [_build_lr_scheduler(optimizer) for optimizer in optimizers]
    )
