import numpy
import sys
import torch

sys.path.append("..")
from os import makedirs
from os.path import exists
from sklearn.metrics import top_k_accuracy_score
from lib.loss import (
	SuperLoss,
	HardFirstSuperLoss,
	MediumFirstSuperLoss,
	TwoEndsFirstSuperLoss,
)

from lib.nn_fns import train_superloss_one_epoch
from lib.util import add_args_str, get_argparser
from baseline.lib.data import get_loader, get_transforms
from baseline.lib.model import (
	CCT_Classifier,
	EfficientNet_Classifier,
	ResNet9_Classifier,
	ResNet_Classifier,
	SERes2Net_Classifier,
	WideResNet_Classifier,
)
from baseline.lib.nn_fns import inference
from baseline.lib.util import save_model


### Option ###

base_model_name = "superloss"
num_classes = 10
train_fn = train_superloss_one_epoch
test_fn = inference

##############


def main() -> None:
	# Arguments
	parser = get_argparser()
	args = parser.parse_args()

	# Setting
	numpy.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	assert torch.cuda.is_available()
	torch.backends.cudnn.deterministic = True
	device = torch.device(f"cuda:{args.gpu_id}")

	# Workspace
	model_name = add_args_str(base_model_name, args)
	model_dir = f"./model/{model_name}/seed{args.seed}"

	if exists(model_dir):
		# Don't run
		print(f"{model_dir} already exists. Skip.")
		return
	else:
		# Run
		makedirs(model_dir)
		print(f"Start: {model_dir}")

	# Load data
	loader_train = get_loader(
		root=args.data_dir,
		train=True,
		transform=get_transforms(train=True, augment=args.augment, jitter=args.jitter),
		download=True,
		batch_size=args.batch_size_train,
		corrupt_label=args.corrupt_label,
	)
	loader_test = get_loader(
		root=args.data_dir,
		train=False,
		transform=get_transforms(train=False),
		download=True,
		batch_size=args.batch_size_test,
	)

	# Create network
	if args.model_type == "cct":
		network = CCT_Classifier(num_classes=num_classes)
	elif args.model_type == "efficientnet":
		network = EfficientNet_Classifier(num_classes=num_classes)
	elif args.model_type == "resnet9":
		network = ResNet9_Classifier(num_classes=num_classes)
	elif args.model_type == "resnet18":
		network = ResNet_Classifier(num_classes=num_classes)
	elif args.model_type == "se_res2net":
		network = SERes2Net_Classifier(num_classes=num_classes)
	elif args.model_type == "wide_resnet":
		network = WideResNet_Classifier(num_classes=num_classes)
	else:
		raise NotImplementedError()
	network.to(device)

	# Loss & Optimizer
	criterion = torch.nn.CrossEntropyLoss(reduction="none")
	optimizer = torch.optim.RAdam(
		network.parameters(), lr=args.lr, weight_decay=args.weight_decay
	)

	# Superloss
	superloss_class = {
		"ef": SuperLoss,
		"hf": HardFirstSuperLoss,
		"mf": MediumFirstSuperLoss,
		"tef": TwoEndsFirstSuperLoss,
	}[args.sl_mode]
	superloss = superloss_class(args.tau, args.lam, args.mom, args.mu).to(device)

	# Run
	for i in range(1, args.epoch + 1):
		# Train
		train_fn(network, loader_train, optimizer, criterion, superloss, i, device)

		# Evaluate
		pred, target = inference(network, loader_test, device)
		acc1 = top_k_accuracy_score(target, pred, k=1)
		print(f"\tacc1: {acc1:.4f}")

		# Save model
		save_model(f"{model_dir}/{i}", network)

		# Write log
		with open(f"{model_dir}/log.txt", "a") as f:
			f.write(f"{i}\t{acc1}\n")

	print(f"Done {model_dir}")


if __name__ == "__main__":
	main()
