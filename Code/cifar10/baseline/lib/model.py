import torch
from typing import Any
from .cct import cct_4_3x2_32
from .efficientnet import efficientnet_b7
from .resnet import resnet18, ResNet9
from .se_res2net import se_res2net50_v1b_26w4s
from .wide_resnet import WideResNet28_10


class CCT_Classifier(torch.nn.Module):
	def __init__(self, num_classes: int, **kwargs: Any) -> None:
		super(CCT_Classifier, self).__init__()
		self.backbone = cct_4_3x2_32()
		self.classifier = torch.nn.Linear(
			self.backbone.encoder.embedding_dim, num_classes
		)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = self.backbone(input)
		logit = self.classifier(x)
		return logit


class EfficientNet_Classifier(torch.nn.Module):
	def __init__(self, num_classes: int, **kwargs: Any) -> None:
		super(EfficientNet_Classifier, self).__init__()
		self.backbone = efficientnet_b7()
		self.classifier = torch.nn.Linear(
			self.backbone.lastconv_output_channels, num_classes
		)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = self.backbone(input)
		logit = self.classifier(x)
		return logit


class ResNet9_Classifier(torch.nn.Module):
	# https://www.kaggle.com/code/kmldas/cifar10-resnet-90-accuracy-less-than-5-min
	def __init__(self, num_classes: int, **kwargs: Any) -> None:
		super(ResNet9_Classifier, self).__init__()
		self.backbone = ResNet9()
		self.classifier = torch.nn.Linear(self.backbone.inplanes, num_classes)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = self.backbone(input)
		logit = self.classifier(x)
		return logit


class ResNet_Classifier(torch.nn.Module):
	def __init__(self, num_classes: int, **kwargs: Any) -> None:
		super(ResNet_Classifier, self).__init__()
		self.backbone = resnet18()
		self.classifier = torch.nn.Linear(self.backbone.inplanes, num_classes)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = self.backbone(input)
		logit = self.classifier(x)
		return logit


class SERes2Net_Classifier(torch.nn.Module):
	def __init__(self, num_classes: int, **kwargs: Any) -> None:
		super(SERes2Net_Classifier, self).__init__()
		self.backbone = se_res2net50_v1b_26w4s()
		self.classifier = torch.nn.Linear(self.backbone.inplanes, num_classes)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = self.backbone(input)
		logit = self.classifier(x)
		return logit


class WideResNet_Classifier(torch.nn.Module):
	def __init__(self, num_classes: int, **kwargs: Any) -> None:
		super(WideResNet_Classifier, self).__init__()
		self.backbone = WideResNet28_10()
		self.classifier = torch.nn.Linear(self.backbone.nStages[-1], num_classes)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		x = self.backbone(input)
		logit = self.classifier(x)
		return logit
