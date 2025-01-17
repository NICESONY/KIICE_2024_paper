import argparse
import time
import torch
from os import rename
from os.path import exists


def get_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()

	parser.add_argument("--gpu_id", type=int, default=0)
	parser.add_argument("--seed", type=int, default=0)

	parser.add_argument("--data_dir", type=str, required=True)

	parser.add_argument("--model_type", type=str, default="")

	parser.add_argument("--augment", action="store_true", default=False)
	parser.add_argument("--jitter", type=float, default=0)
	parser.add_argument("--corrupt_label", type=float, default=0)

	parser.add_argument("--batch_size_train", type=int, default=128)
	parser.add_argument("--batch_size_test", type=int, default=512)

	parser.add_argument("--lr", type=float, default=1e-03)
	parser.add_argument("--weight_decay", type=float, default=1e-05)
	parser.add_argument("--epoch", type=int, default=100)

	return parser


def add_args_str(base_model_name: str, args: argparse.Namespace) -> str:
	model_name = f"{base_model_name}_{args.model_type}"
	if args.augment:
		model_name += "_aug"
	if args.jitter > 0:
		model_name += f"_jitter_{args.jitter}"
	if 0 < args.corrupt_label < 1:
		model_name += f"_corrupt_label_{args.corrupt_label}"
	return model_name


def save_model(model_path: str, network: torch.nn.Module) -> None:
	if exists(model_path):
		timestamp_str = time.strftime("%Y%m%d_%H%M%S")
		print(f" Already exists: {model_path}. Move it to {model_path}_{timestamp_str}")
		rename(model_path, f"{model_path}_{timestamp_str}")

	torch.save(network.state_dict(), model_path)
