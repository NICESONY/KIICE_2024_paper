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
	"tau": 2.3,  # 2.3 for CIFAR-10;	4.6 for CIFAR-100
	"mom": 0.1,  # Fixed at 0.1
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

	for jitter in [0.0]:  # , 0.1, 0.01]:
		for corrupt_label in [0.0, 0.1, 0.3, 0.5, 0.7]:
			for sl_mode1, sl_mode2 in [
				("ef", "hf"),
				("ef", "mf"),
			]:
				for lam in [0.25, 0.5, 1.0]:
					for change_epoch in [10, 20, 30, 50]:
						run(
							f"{cmd} --sl_mode1 {sl_mode1} --sl_mode2 {sl_mode2} --lam {lam} --jitter {jitter} --corrupt_label {corrupt_label} --change_epoch {change_epoch}",
							cwd=f"./mode_changable_superloss",
							shell=True,
						)


def main() -> None:
	run_per_seed(0)


if __name__ == "__main__":
	main()
