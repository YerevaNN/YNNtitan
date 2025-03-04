# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import submitit


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="~/slurm_jobs/titan/job_%j")
    node = "h100"
    n_gpus = 6
    experiment_name = "3d_drugs_cart_2E_lr4e-3"
    # executor = submitit.local.local.LocalExecutor(folder="~/slurm_jobs/titan/job_%j")
    # node = "local"
    # n_gpus = 1
    # experiment_name = "conformer_conversion"
    executor.update_parameters(
        name=experiment_name,
        timeout_min=24 * 24 * 60,
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=80,
        cpus_per_task=n_gpus * 18,
        slurm_additional_parameters={"partition": node},
    )

    jobs = []
    with executor.batch():
        for _ in range(1):
            # train_config = './train_configs/chemlactica_125m.toml'
            # train_config = './train_configs/chemlactica_1.3b.toml'
            # train_config = "./train_configs/llama3.2_1b.toml"
            train_config = "./train_configs/llama3.2_1b_conformers.toml"
            # train_config = "./train_configs/llama3.2_1b_conversion.toml"
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
                    "--metrics.aim_experiment_name",
                    experiment_name,
                    # "--training.num_decays",
                    # "6",
                    # "--training.decay_steps_perc",
                    # "0.1",
                ]
            )
            print(" ".join(function.command))
            # subprocess.run(function.command)
            job = executor.submit(function)
            jobs.append(job)
