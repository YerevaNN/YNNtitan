# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import submitit


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="~/slurm_jobs/titan/job_%j")
    n_gpus = 6
    node = "h100"
    executor.update_parameters(
        name="titan",
        timeout_min=6 * 60,
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=80,
        cpus_per_task=n_gpus * 12,
        slurm_additional_parameters={"partition": node},
    )

    hparams = {
        # "optimizer.lr": ["1.2e-3", "9e-4", "6e-4", "3e-4"],
        # "optimizer.lr": ["8e-4", "6e-4", "4e-4", "2e-4"],
        # "optimizer.lr": ["2.5e-4"],
        # "optimizer.lr": ["1e-4", "8e-5", "6e-5", "4e-5", "2e-5"],
        "training.gradient_accumulation_steps": ["21", "25", "29", "33"],
        "training.steps": ["31000", "26000", "22.500", "20000"],
    }

    jobs = []
    with executor.batch():
        for _ in range(1):
            length = len(list(hparams.values())[0])
            for i in range(length):
                hparam_dict = {}
                for key, values in hparams.items():
                    hparam_dict[key] = values[i]

                # train_config = './train_configs/chemlactica_125m.toml'
                # train_config = './train_configs/chemlactica_1.3b.toml'
                train_config = "./train_configs/llama3.2_1b.toml"
                # train_config = './train_configs/debug_model.toml'
                command_lst = [
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
                ]

                # add the hparam
                for key, value in hparam_dict.items():
                    command_lst.append(f"--{key}")
                    command_lst.append(value)

                function = submitit.helpers.CommandFunction(command_lst)
                print(" ".join(function.command))
                # subprocess.run(function.command)
                job = executor.submit(function)
                jobs.append(job)
