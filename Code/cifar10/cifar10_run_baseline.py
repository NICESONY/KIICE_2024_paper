from subprocess import run


### Option ###
gpu_id = 0
seed = 0

python_str = "python3"

args_lstm = {
    "gpu_id": gpu_id,
    "seed": seed,
    "data_dir": "./",  ########## Please modify as you want
    "model_type": ["cct", "se_res2net"][1],
    "augment": False,
    "batch_size_train": 128,
    "batch_size_test": 256,
    "lr": 5e-04,
    "weight_decay": 1e-05,
    "epoch": 100,
}
##############


def run_per_seed(seed: int) -> None:
    script_name = f"main.py"
    args_str = " ".join(
        f"--{k}" if isinstance(v, bool) and v else f"--{k} {v}"
        for k, v in args_lstm.items()
        if not (isinstance(v, bool) and (not v))
    )

    cmd = f"{python_str} {script_name} {args_str} --seed {seed}"

    for jitter in [0.0]:#, 0.1, 0.01]:
        for corrupt_label in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]:
            run(
                f"{cmd} --jitter {jitter} --corrupt_label {corrupt_label}",
                cwd=f"./baseline",
                shell=True,
            )


def main() -> None:
    run_per_seed(0)


if __name__ == "__main__":
    main()
