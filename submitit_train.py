# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import submitit


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="~/slurm_jobs/titan/job_%j")
    n_gpus = 8
    node = "h100"
    executor.update_parameters(
        name="titan",
        timeout_min=24 * 24 * 60,
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=80,
        cpus_per_task=n_gpus * 12,
        slurm_additional_parameters={"partition": node},
    )

    jobs = []
    with executor.batch():
        for _ in range(1):
            # train_config = './train_configs/chemlactica_125m.toml'
            # train_config = './train_configs/chemlactica_1.3b.toml'
            # train_config = "./train_configs/llama3_125m.toml"
            train_config = "./train_configs/llama3.2_1b.toml"
            # train_config = "./train_configs/llama3.2_3b.toml"
            # train_config = './train_configs/debug_model.toml'
            function = submitit.helpers.CommandFunction(
                [
                    "python3",
                    "-m",
                    "torch.distributed.run",
                    "--nproc_per_node",
                    f"{n_gpus}",
                    "--rdzv_backend",
                    "c10d",
                    "--rdzv_endpoint",
                    "localhost:0",
                    "--local-ranks-filter",
                    "0",
                    "--role",
                    "rank",
                    "--tee",
                    "3",
                    "train.py",
                    "--job.config_file",
                    train_config,
                    "--training.steps",
                    "20000",
                ]
            )
            print(" ".join(function.command))
            # subprocess.run(function.command)
            job = executor.submit(function)
            jobs.append(job)
