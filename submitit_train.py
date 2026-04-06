# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import submitit
import os


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder=f"{os.environ['LOG_DIR']}/slurm_logs/titan/job_%j")
    n_gpus = 6
    node = "all"
    executor.update_parameters(
        name="titan",
        timeout_min=2 * 24 * 60,
        gpus_per_node=n_gpus,
        nodes=1,
        mem_gb=200,
        cpus_per_task=184,
        slurm_additional_parameters={"partition": node},
        # Compute nodes often lack srun on PATH for non-login batch scripts; single-node
        # jobs do not need an srun step—Slurm already placed the allocation.
        use_srun=False,
    )

    jobs = []
    with executor.batch():
        for _ in range(1):
            # train_config = './train_configs/chemlactica_125m.toml'
            # train_config = './train_configs/chemlactica_1.3b.toml'
            train_config = "./train_configs/llama3_170m.toml"
            #train_config = "./train_configs/llama3_380m.toml"
            # train_config = "./train_configs/llama3_750m.toml"
            # train_config = "./train_configs/llama3_1b_pubchem.toml"
            # train_config = "./train_configs/llama3_380m_pubchem.toml"
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
                ]
            )
            print(" ".join(function.command))
            # subprocess.run(function.command)
            job = executor.submit(function)
            jobs.append(job)
