import numpy
import random
import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms
from torchvision.datasets import CIFAR10
from typing import Callable, Optional, Tuple


class ShearX(object):
	def __init__(self, fillcolor=(128, 128, 128)):
		self.fillcolor = fillcolor

	def __call__(self, x, magnitude):
		return x.transform(
			x.size,
			Image.AFFINE,
			(1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
			Image.BICUBIC,
			fillcolor=self.fillcolor,
		)


class ShearY(object):
	def __init__(self, fillcolor=(128, 128, 128)):
		self.fillcolor = fillcolor

	def __call__(self, x, magnitude):
		return x.transform(
			x.size,
			Image.AFFINE,
			(1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
			Image.BICUBIC,
			fillcolor=self.fillcolor,
		)


class TranslateX(object):
	def __init__(self, fillcolor=(128, 128, 128)):
		self.fillcolor = fillcolor

	def __call__(self, x, magnitude):
		return x.transform(
			x.size,
			Image.AFFINE,
			(1, 0, magnitude * x.size[0] * random.choice([-1, 1]), 0, 1, 0),
			fillcolor=self.fillcolor,
		)


class TranslateY(object):
	def __init__(self, fillcolor=(128, 128, 128)):
		self.fillcolor = fillcolor

	def __call__(self, x, magnitude):
		return x.transform(
			x.size,
			Image.AFFINE,
			(1, 0, 0, 0, 1, magnitude * x.size[1] * random.choice([-1, 1])),
			fillcolor=self.fillcolor,
		)


class Rotate(object):
	# from https://stackoverflow.com/questions/
	# 5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
	def __call__(self, x, magnitude):
		rot = x.convert("RGBA").rotate(magnitude)
		return Image.composite(
			rot, Image.new("RGBA", rot.size, (128,) * 4), rot
		).convert(x.mode)


class Color(object):
	def __call__(self, x, magnitude):
		return ImageEnhance.Color(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Posterize(object):
	def __call__(self, x, magnitude):
		return ImageOps.posterize(x, magnitude)


class Solarize(object):
	def __call__(self, x, magnitude):
		return ImageOps.solarize(x, magnitude)


class Contrast(object):
	def __call__(self, x, magnitude):
		return ImageEnhance.Contrast(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Sharpness(object):
	def __call__(self, x, magnitude):
		return ImageEnhance.Sharpness(x).enhance(1 + magnitude * random.choice([-1, 1]))


class Brightness(object):
	def __call__(self, x, magnitude):
		return ImageEnhance.Brightness(x).enhance(
			1 + magnitude * random.choice([-1, 1])
		)


class AutoContrast(object):
	def __call__(self, x, magnitude):
		return ImageOps.autocontrast(x)


class Equalize(object):
	def __call__(self, x, magnitude):
		return ImageOps.equalize(x)


class Invert(object):
	def __call__(self, x, magnitude):
		return ImageOps.invert(x)


class SubPolicy(object):
	def __init__(
		self,
		p1,
		operation1,
		magnitude_idx1,
		p2,
		operation2,
		magnitude_idx2,
		fillcolor=(128, 128, 128),
	):
		ranges = {
			"shearX": numpy.linspace(0, 0.3, 10),
			"shearY": numpy.linspace(0, 0.3, 10),
			"translateX": numpy.linspace(0, 150 / 331, 10),
			"translateY": numpy.linspace(0, 150 / 331, 10),
			"rotate": numpy.linspace(0, 30, 10),
			"color": numpy.linspace(0.0, 0.9, 10),
			"posterize": numpy.round(numpy.linspace(8, 4, 10), 0).astype(numpy.int),
			"solarize": numpy.linspace(256, 0, 10),
			"contrast": numpy.linspace(0.0, 0.9, 10),
			"sharpness": numpy.linspace(0.0, 0.9, 10),
			"brightness": numpy.linspace(0.0, 0.9, 10),
			"autocontrast": [0] * 10,
			"equalize": [0] * 10,
			"invert": [0] * 10,
		}

		func = {
			"shearX": ShearX(fillcolor=fillcolor),
			"shearY": ShearY(fillcolor=fillcolor),
			"translateX": TranslateX(fillcolor=fillcolor),
			"translateY": TranslateY(fillcolor=fillcolor),
			"rotate": Rotate(),
			"color": Color(),
			"posterize": Posterize(),
			"solarize": Solarize(),
			"contrast": Contrast(),
			"sharpness": Sharpness(),
			"brightness": Brightness(),
			"autocontrast": AutoContrast(),
			"equalize": Equalize(),
			"invert": Invert(),
		}

		self.p1 = p1
		self.operation1 = func[operation1]
		self.magnitude1 = ranges[operation1][magnitude_idx1]
		self.p2 = p2
		self.operation2 = func[operation2]
		self.magnitude2 = ranges[operation2][magnitude_idx2]

	def __call__(self, img):
		if random.random() < self.p1:
			img = self.operation1(img, self.magnitude1)
		if random.random() < self.p2:
			img = self.operation2(img, self.magnitude2)
		return img


class CIFAR10Policy(object):
	"""Randomly choose one of the best 25 Sub-policies on CIFAR10.

	Example:
	>>> policy = CIFAR10Policy()
	>>> transformed = policy(image)

	Example as a PyTorch Transform:
	>>> transform=transforms.Compose([
	>>>     transforms.Resize(256),
	>>>     CIFAR10Policy(),
	>>>     transforms.ToTensor()])
	"""

	def __init__(self, fillcolor=(128, 128, 128)):
		self.policies = [
			SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
			SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
			SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
			SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
			SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
			SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
			SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
			SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
			SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
			SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
			SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
			SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
			SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
			SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
			SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
			SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
			SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
			SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
			SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
			SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
			SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
			SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
			SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
			SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
			SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
		]

	def __call__(self, img):
		policy_idx = random.randint(0, len(self.policies) - 1)
		return self.policies[policy_idx](img)

	def __repr__(self):
		return "AutoAugment CIFAR10 Policy"


class RandomJitter(torch.nn.Module):
	def __init__(self, p: float = 0.5, std: float = 0.01):
		super().__init__()
		self.p = p
		self.std = std

	def forward(self, img):
		if torch.rand(1) < self.p:
			return img + torch.normal(
				mean=0, std=self.std, size=img.shape, dtype=torch.float
			)
		return img

	def __repr__(self) -> str:
		return f"{self.__class__.__name__}(p={self.p})"


def get_transforms(
	train: bool,
	mean: Tuple = (0.4914, 0.4822, 0.4465),
	std: Tuple = (0.2470, 0.2435, 0.2616),
	augment: bool = False,
	jitter: float = 0,
) -> Callable:
	if train:
		if augment:
			transform_sequence = [CIFAR10Policy()]
		else:
			transform_sequence = []

		transform_sequence += [
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean, std, inplace=True),
		]

		# (Optional) Jittering
		if jitter > 0:
			transform_sequence.append(RandomJitter(std=jitter))
	else:
		transform_sequence = [
			transforms.ToTensor(),
			transforms.Normalize(mean, std, inplace=True),
		]

	return transforms.Compose(transform_sequence)


def get_loader(
	root: str,
	train: bool,
	transform: Optional[Callable] = None,
	target_transform: Optional[Callable] = None,
	download: bool = False,
	corrupt_label: float = 0,
	batch_size: int = 128,
) -> torch.utils.data.DataLoader:
	dataset = CIFAR10(
		root,
		train=train,
		transform=transform,
		target_transform=target_transform,
		download=download,
	)

	if train:
		# (Optional) Label corruption
		if 0 < corrupt_label <= 1:
			# https://github.com/apple/ml-data-parameters/blob/main/dataset/cifar_dataset.py#L34

			# Static generator that does not depend on the seeds manually set by user
			generator = numpy.random.default_rng(1108)

			N = len(dataset)
			S = int(N * corrupt_label)
			assert 0 < S < N

			idx_N_shuffled = generator.permutation(N)
			data = dataset.data[idx_N_shuffled]
			targets = numpy.asarray(dataset.targets)[idx_N_shuffled]

			targets_corrupted = targets[:S]
			targets_clean = targets[S:]

			idx_S_shuffled = generator.permutation(S)
			targets_corrupted = targets_corrupted[idx_S_shuffled]
			targets = numpy.concatenate(
				[targets_corrupted, targets_clean], axis=0
			).astype(dtype=numpy.int64)

			dataset.data = data
			dataset.targets = list(targets)

		#
		loader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=True,
			pin_memory=True,
			drop_last=False,
		)
	else:
		loader = torch.utils.data.DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=False,
			pin_memory=True,
			drop_last=False,
		)

	return loader
