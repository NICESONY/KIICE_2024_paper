import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Any, Callable, List, Optional, Type


class SqueezeExcitation(nn.Module):
	# torchvision.ops.misc.SqueezeExcitation
	def __init__(
		self,
		input_channels: int,
		squeeze_channels: int,
		bias: bool = True,
		activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
		scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
	) -> None:
		super().__init__()
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1, bias=bias)
		self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1, bias=bias)
		self.activation = activation()
		self.scale_activation = scale_activation()

	def _scale(self, input: torch.Tensor) -> torch.Tensor:
		scale = self.avgpool(input)
		scale = self.fc1(scale)
		scale = self.activation(scale)
		scale = self.fc2(scale)
		return self.scale_activation(scale)

	def forward(self, input: torch.Tensor) -> torch.Tensor:
		scale = self._scale(input)
		return scale * input


class SEBottle2neck(nn.Module):
	def __init__(
		self,
		inplanes: int,
		planes: int,
		stride: int = 1,
		downsample: Optional[nn.Module] = None,
		groups: int = 1,
		base_width: int = 26,
		dilation: int = 1,
		norm_layer: Optional[Callable[..., nn.Module]] = None,
		scale: int = 4,
		stype: str = "normal",
		expansion: int = 4,
		activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
		squeeze_ratio: int = 16,
	) -> None:
		super(SEBottle2neck, self).__init__()

		self.scale = int(scale)
		self.stype = str(stype)
		self.width = int(planes * (base_width / 64.0)) * groups

		if norm_layer is None:
			norm_layer = nn.BatchNorm2d

		self.conv1 = nn.Conv2d(inplanes, self.width * self.scale, 1, bias=False)
		self.bn1 = norm_layer(self.width * self.scale)

		if self.stype == "stage":
			self.pool = nn.AvgPool2d(3, stride=stride, padding=1)

		self.nums = 1 if self.scale == 1 else self.scale - 1
		self.conv2 = nn.ModuleList(
			[
				nn.Conv2d(
					self.width,
					self.width,
					3,
					stride=stride,
					padding=1,
					dilation=dilation,
					groups=groups,
					bias=False,
				)
				for _ in range(self.nums)
			]
		)
		self.bn2 = nn.ModuleList([norm_layer(self.width) for _ in range(self.nums)])

		self.conv3 = nn.Conv2d(
			self.width * self.scale, planes * expansion, 1, bias=False
		)
		self.bn3 = norm_layer(planes * expansion)

		self.act = activation()
		self.se = SqueezeExcitation(
			planes * expansion,
			(planes * expansion) // squeeze_ratio,
			bias=True,
			activation=activation,
		)
		self.downsample = downsample

	def forward(self, x: Tensor) -> Tensor:
		identity = x if self.downsample is None else self.downsample(x)

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.act(out)

		spx = torch.split(out, self.width, dim=1)
		for i in range(self.nums):
			if i == 0 or self.stype == "stage":
				sp = spx[i]
			else:
				sp = sp + spx[i]

			sp = self.conv2[i](sp)
			sp = self.bn2[i](sp)
			sp = self.act(sp)

			if i == 0:
				out = sp
			else:
				out = torch.cat((out, sp), dim=1)

		if self.scale != 1:
			if self.stype == "normal":
				out = torch.cat((out, spx[self.nums]), dim=1)
			elif self.stype == "stage":
				out = torch.cat((out, self.pool(spx[self.nums])), dim=1)

		out = self.conv3(out)
		out = self.bn3(out)

		out = self.se(out) + identity
		out = self.act(out)

		return out


class SERes2Net(nn.Module):
	def __init__(
		self,
		block: Type[SEBottle2neck],
		layers: List[int],
		num_input_channels: int = 3,
		stem_planes: List[int] = [16, 16, 32],#[32, 32, 64],
		stem_strides: List[int] = [2, 1, 1],
		zero_init_residual: bool = False,
		groups: int = 1,
		width_per_group: int = 26,
		replace_stride_with_dilation: Optional[List[bool]] = None,
		norm_layer: Optional[Callable[..., nn.Module]] = None,
		scale: int = 4,
		expansion: int = 2,#4,
		activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
		squeeze_ratio: int = 16,
		dropout: float = 0,
		**kwargs: Any
	) -> None:
		super(SERes2Net, self).__init__()
		self._norm_layer = nn.BatchNorm2d if norm_layer is None else norm_layer

		assert len(stem_planes) == len(stem_strides) == 3
		self.base_width = int(width_per_group)
		self.dilation = 1
		self.expansion = int(expansion)
		self.groups = int(groups)
		self.inplanes = int(stem_planes[2])
		self.scale = int(scale)
		self.squeeze_ratio = int(squeeze_ratio)

		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError(
				"replace_stride_with_dilation should be None "
				"or a 3-element tuple, got {}".format(replace_stride_with_dilation)
			)

		self.stem = nn.Sequential(
			nn.Conv2d(
				num_input_channels,
				stem_planes[0],
				3,
				stride=stem_strides[0],
				padding=1,
				bias=False,
			),
			self._norm_layer(stem_planes[0]),
			activation(),
			nn.Conv2d(
				stem_planes[0],
				stem_planes[1],
				3,
				stride=stem_strides[1],
				padding=1,
				bias=False,
			),
			self._norm_layer(stem_planes[1]),
			activation(),
			nn.Conv2d(
				stem_planes[1],
				stem_planes[2],
				3,
				stride=stem_strides[2],
				padding=1,
				bias=False,
			),
			self._norm_layer(stem_planes[2]),
			activation(),
			nn.MaxPool2d(3, stride=2, padding=1),
		)

		self.layer1 = self._make_layer(
			block, stem_planes[2], layers[0], activation=activation
		)
		assert self.inplanes == stem_planes[2] * self.expansion

		self.layer2 = self._make_layer(
			block,
			stem_planes[2] * 2,
			layers[1],
			stride=2,
			dilate=replace_stride_with_dilation[0],
			activation=activation,
		)
		assert self.inplanes == stem_planes[2] * self.expansion * 2

		self.layer3 = self._make_layer(
			block,
			stem_planes[2] * 4,
			layers[2],
			stride=2,
			dilate=replace_stride_with_dilation[1],
			activation=activation,
		)
		assert self.inplanes == stem_planes[2] * self.expansion * 4

		self.layer4 = self._make_layer(
			block,
			stem_planes[2] * 8,
			layers[3],
			stride=2,
			dilate=replace_stride_with_dilation[2],
			activation=activation,
		)
		assert self.inplanes == stem_planes[2] * self.expansion * 8

		self.avgpool = nn.Sequential(
			nn.AdaptiveAvgPool2d(1), nn.Dropout(p=dropout, inplace=True)
		)
		"""
		for m in self.modules():
			if isinstance(m, (nn.Conv2d, nn.Linear)):
				fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
				nn.init.trunc_normal_(m.weight, std=math.sqrt(1.0 / fan_in))
				if m.bias is not None:
					nn.init.zeros_(m.bias)
		"""
		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, SEBottle2neck):
					# type: ignore[arg-type]
					nn.init.zeros_(m.bn3.weight)

	def _make_layer(
		self,
		block: Type[SEBottle2neck],
		planes: int,
		blocks: int,
		stride: int = 1,
		dilate: bool = False,
		activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
	) -> nn.Sequential:
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * self.expansion:
			downsample = nn.Sequential(
				nn.AvgPool2d(
					stride, stride=stride, ceil_mode=True, count_include_pad=False
				),
				nn.Conv2d(
					self.inplanes, planes * self.expansion, 1, stride=1, bias=False
				),
				self._norm_layer(planes * self.expansion),
			)

		layers = []
		layers.append(
			block(
				self.inplanes,
				planes,
				stride=stride,
				downsample=downsample,
				groups=self.groups,
				base_width=self.base_width,
				dilation=previous_dilation,
				norm_layer=self._norm_layer,
				scale=self.scale,
				stype="stage",
				expansion=self.expansion,
				activation=activation,
				squeeze_ratio=self.squeeze_ratio,
			)
		)
		self.inplanes = planes * self.expansion
		for _ in range(1, blocks):
			layers.append(
				block(
					self.inplanes,
					planes,
					groups=self.groups,
					base_width=self.base_width,
					dilation=self.dilation,
					norm_layer=self._norm_layer,
					scale=self.scale,
					expansion=self.expansion,
					activation=activation,
					squeeze_ratio=self.squeeze_ratio,
				)
			)

		return nn.Sequential(*layers)

	def _compute_fmap(self, x: Tensor) -> Tensor:
		x = self.stem(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x

	def forward(self, x: Tensor) -> Tensor:
		x = self._compute_fmap(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		return x


def _se_res2net(
	arch: str, block: Type[SEBottle2neck], layers: List[int], **kwargs: Any
) -> SERes2Net:
	model = SERes2Net(block, layers, **kwargs)
	return model


def se_res2net50_v1b_26w4s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 26
	kwargs["scale"] = 4
	return _se_res2net("se_res2net50_v1b_26w4s", SEBottle2neck, [3, 4, 6, 3], **kwargs)


def se_res2net50_v1b_26w6s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 26
	kwargs["scale"] = 6
	return _se_res2net("se_res2net50_v1b_26w6s", SEBottle2neck, [3, 4, 6, 3], **kwargs)


def se_res2net50_v1b_26w8s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 26
	kwargs["scale"] = 8
	return _se_res2net("se_res2net50_v1b_26w8s", SEBottle2neck, [3, 4, 6, 3], **kwargs)


def se_res2net50_v1b_14w8s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 14
	kwargs["scale"] = 8
	return _se_res2net("se_res2net50_v1b_14w8s", SEBottle2neck, [3, 4, 6, 3], **kwargs)


def se_res2net50_v1b_48w2s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 48
	kwargs["scale"] = 2
	return _se_res2net("se_res2net50_v1b_48w2s", SEBottle2neck, [3, 4, 6, 3], **kwargs)


def se_res2net101_v1b_26w4s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 26
	kwargs["scale"] = 4
	return _se_res2net(
		"se_res2net101_v1b_26w4s", SEBottle2neck, [3, 4, 23, 3], **kwargs
	)


def se_res2net101_v1b_26w6s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 26
	kwargs["scale"] = 6
	return _se_res2net(
		"se_res2net101_v1b_26w6s", SEBottle2neck, [3, 4, 23, 3], **kwargs
	)


def se_res2net101_v1b_26w8s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 26
	kwargs["scale"] = 8
	return _se_res2net(
		"se_res2net101_v1b_26w8s", SEBottle2neck, [3, 4, 23, 3], **kwargs
	)


def se_res2net101_v1b_14w8s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 14
	kwargs["scale"] = 8
	return _se_res2net(
		"se_res2net101_v1b_14w8s", SEBottle2neck, [3, 4, 23, 3], **kwargs
	)


def se_res2net101_v1b_48w2s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 48
	kwargs["scale"] = 2
	return _se_res2net(
		"se_res2net101_v1b_48w2s", SEBottle2neck, [3, 4, 23, 3], **kwargs
	)


def se_res2net152_v1b_26w4s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 26
	kwargs["scale"] = 4
	return _se_res2net(
		"se_res2net152_v1b_26w4s", SEBottle2neck, [3, 8, 36, 3], **kwargs
	)


def se_res2net152_v1b_26w6s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 26
	kwargs["scale"] = 6
	return _se_res2net(
		"se_res2net152_v1b_26w6s", SEBottle2neck, [3, 8, 36, 3], **kwargs
	)


def se_res2net152_v1b_26w8s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 26
	kwargs["scale"] = 8
	return _se_res2net(
		"se_res2net152_v1b_26w8s", SEBottle2neck, [3, 8, 36, 3], **kwargs
	)


def se_res2net152_v1b_14w8s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 14
	kwargs["scale"] = 8
	return _se_res2net(
		"se_res2net152_v1b_14w8s", SEBottle2neck, [3, 8, 36, 3], **kwargs
	)


def se_res2net152_v1b_48w2s(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 48
	kwargs["scale"] = 2
	return _se_res2net(
		"se_res2net152_v1b_48w2s", SEBottle2neck, [3, 8, 36, 3], **kwargs
	)


def se_res2next50_v1b_26w4s8x(**kwargs: Any) -> SERes2Net:
	kwargs["width_per_group"] = 26
	kwargs["scale"] = 4
	kwargs["groups"] = 8
	return _se_res2net(
		"se_res2next50_v1b_26w4s8x", SEBottle2neck, [3, 4, 6, 3], **kwargs
	)
