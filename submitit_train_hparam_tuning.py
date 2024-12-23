import submitit
import datetime
import yaml
import os


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="~/slurm_jobs/titan/job_%j")
    n_gpus = 8
    executor.update_parameters(
        name="titan", timeout_min=3 * 60,
        gpus_per_node=n_gpus,
        nodes=1, mem_gb=80, cpus_per_task=n_gpus * 4,
        slurm_additional_parameters={
            "partition": "h100"
        }
    )

    hparams = {
        # "optimizer.lr": ["1.2e-3", "9e-4", "6e-4", "3e-4"],
        # "optimizer.lr": ["8e-4", "6e-4", "4e-4", "2e-4"],
        # "optimizer.lr": ["2.5e-4"],
        # "optimizer.lr": ["1e-4", "8e-5", "6e-5", "4e-5", "2e-5"],
    }

    jobs = []
    with executor.batch():
        for _ in range(1):
            for hparam_name, value in hparams.items():
                for v in value:
                    # train_config = './train_configs/chemlactica_125m.toml'
                    # train_config = './train_configs/chemlactica_1.3b.toml'
                    train_config = './train_configs/llama3.2_1b.toml'
                    # train_config = './train_configs/debug_model.toml'
                    function = submitit.helpers.CommandFunction([
                        'python3', '-m', 'torch.distributed.run',
                        '--nproc_per_node', f'{n_gpus}',
                        '--rdzv_backend', 'c10d',
                        '--rdzv_endpoint', 'localhost:0',
                        '--local-ranks-filter', '0',
                        '--role', 'rank', '--tee', '3',
                        'train.py',
                        '--job.config_file', train_config,
                        f'--{hparam_name}', v
                    ])
                    print(' '.join(function.command))
                    # subprocess.run(function.command)
                    job = executor.submit(function)
                    jobs.append(job)
