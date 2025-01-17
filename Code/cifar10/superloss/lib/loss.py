import math
import numpy
import torch
import torch.nn as nn
from typing import Optional, Tuple
from scipy.special import lambertw


class AdditiveAngluarMargin(nn.Module):
	def __init__(
		self, in_features: int, num_classes: int, margin: float = 0.9, scale: float = 20
	) -> None:
		super(AdditiveAngluarMargin, self).__init__()

		_weight_data = torch.empty((num_classes, in_features), dtype=torch.float)
		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(_weight_data)
		nn.init.trunc_normal_(_weight_data, std=math.sqrt(1.0 / fan_in))
		self.weight = nn.Parameter(_weight_data.T)

		self.register_buffer("margin", torch.as_tensor(margin, dtype=torch.float))
		self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float))

	def forward(
		self, input: torch.Tensor, target: Optional[torch.Tensor] = None
	) -> torch.Tensor:
		assert input.ndim == 2
		x = nn.functional.normalize(input, dim=1)
		w = nn.functional.normalize(self.weight, dim=0)

		logit = torch.mm(x, w)
		logit = torch.clamp(logit, -1, 1)  # ?

		if self.training:
			assert (
				target is not None
				and target.ndim == 1
				and logit.shape[0] == target.shape[0]
			)
			batch_arange = torch.arange(input.size(0), dtype=torch.int)

			logit = torch.acos(logit)
			logit[batch_arange, target] = logit[batch_arange, target] + self.margin
			logit = torch.cos(logit) * self.scale
		else:
			assert target is None

		return logit


class P2SGrad(nn.Module):
	def __init__(self, in_features: int, num_classes: int) -> None:
		super(P2SGrad, self).__init__()

		_weight_data = torch.empty((num_classes, in_features), dtype=torch.float)
		fan_in, _ = nn.init._calculate_fan_in_and_fan_out(_weight_data)
		nn.init.trunc_normal_(_weight_data, std=math.sqrt(1.0 / fan_in))
		self.weight = nn.Parameter(_weight_data.T)

	def forward(
		self,
		input: torch.Tensor,
		target: Optional[torch.Tensor] = None,
		weight: Optional[torch.Tensor] = None,
	) -> torch.Tensor:
		assert input.ndim == 2
		x = nn.functional.normalize(input, dim=1)
		w = nn.functional.normalize(self.weight, dim=0)
		logit = torch.mm(x, w)
		logit = torch.clamp(logit, -1, 1)  # ?

		if self.training:
			assert (
				target is not None
				and target.ndim == 1
				and logit.shape[0] == target.shape[0]
			)
			with torch.no_grad():
				offset = torch.zeros_like(logit)
				offset.scatter_(dim=1, index=target.data.view(-1, 1), value=1)

			loss = nn.functional.mse_loss(logit, offset, reduction="none")
			if weight is not None:
				assert (
					isinstance(weight, torch.Tensor)
					and weight.ndim == 1
					and weight.shape[0] == loss.shape[1]
				)
				wsum = weight.sum()
				if wsum != 1:
					weight = weight / wsum
				loss = (loss * weight) * float(weight.shape[0])

			return loss * 0.5

		else:
			assert target is None
			return logit


class SuperLoss(nn.Module):
	# L = (loss - tau) * sigma + lambda * (log(sigma) ** 2)
	# https://github.com/AlanChou/Super-Loss/blob/main/SuperLoss.py
	# https://github.com/THUMNLab/CurML/blob/master/curriculum/algorithms/superloss.py

	def __init__(
		self, lam: float = 1.0, tau: float = 0.5, mom: float = 0.1, mu: float = 0.0
	) -> None:
		super(SuperLoss, self).__init__()
		assert 0 < lam and 0 <= mom <= 1 and 0 <= mu

		self.inv_lam = float(1.0 / lam)  # regularization hparam;	e.g., 1, 0.25
		self.tau = float(
			tau
		)  # threshold;	running average of input loss;	e.g., log(C) in classification
		self.mom = float(mom)  # e.g., 0.1
		self.mu = float(mu)

	def compute_sigma(self, loss: torch.Tensor) -> torch.Tensor:
		loss_numpy = loss.detach().cpu().numpy()

		if self.mom > 0:
			self.tau = (1.0 - self.mom) * self.tau + self.mom * loss_numpy.mean()

		beta = (loss_numpy - self.tau) * self.inv_lam
		z = 0.5 * numpy.maximum(-2.0 * numpy.exp(-1), beta)

		sig = torch.from_numpy(numpy.exp(-lambertw(z).real)).to(
			dtype=loss.dtype, device=loss.device
		)

		return sig

	def forward(self, loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		sigma = self.compute_sigma(loss)
		superloss = sigma * loss
		return superloss, sigma


class HardFirstSuperLoss(SuperLoss):
	# L = (loss - tau) * sigma - lambda * (log(sigma) ** 2)

	def compute_sigma(self, loss: torch.Tensor) -> torch.Tensor:
		loss_numpy = loss.detach().cpu().numpy()

		if self.mom > 0:
			self.tau = (1.0 - self.mom) * self.tau + self.mom * loss_numpy.mean()

		beta = (loss_numpy - self.tau) * self.inv_lam
		z = 0.5 * numpy.minimum(2.0 * numpy.exp(-1), beta)

		sig = torch.from_numpy(numpy.exp(-lambertw(-z).real)).to(
			dtype=loss.dtype, device=loss.device
		)

		return sig


class MediumFirstSuperLoss(SuperLoss):
	# L = (loss - tau) * sigma + sign(loss - tau) * lambda * ((log(sigma) ** 2) + 2 * sigma * (mu - exp(-1)))
	@staticmethod
	def _compute_sigma_internal(
		beta: numpy.ndarray, mu: float, mode: str
	) -> numpy.ndarray:
		if mode == "easy":
			z = 0.5 * beta + numpy.exp(-1) - mu
			sig = numpy.exp(-lambertw(-z).real)
		else:
			assert mode == "hard"
			z = 0.5 * beta - numpy.exp(-1) + mu
			sig = numpy.exp(-lambertw(z).real)

		return sig

	def compute_sigma(self, loss: torch.Tensor) -> torch.Tensor:
		loss_numpy = loss.detach().cpu().numpy()

		if self.mom > 0:
			self.tau = (1.0 - self.mom) * self.tau + self.mom * loss_numpy.mean()

		beta = (loss_numpy - self.tau) * self.inv_lam
		sig = numpy.empty_like(beta)

		idx_easy = beta < 0
		idx_hard = ~idx_easy
		sig[idx_easy] = self._compute_sigma_internal(beta[idx_easy], self.mu, "easy")
		sig[idx_hard] = self._compute_sigma_internal(beta[idx_hard], self.mu, "hard")

		sig = torch.from_numpy(sig).to(dtype=loss.dtype, device=loss.device)

		return sig


class TwoEndsFirstSuperLoss(MediumFirstSuperLoss):
	# L = (loss - tau) * sigma - sign(loss - tau) * lambda * ((log(sigma) ** 2) + 2 * sigma * mu)

	@staticmethod
	def _compute_sigma_internal(
		beta: numpy.ndarray, mu: float, mode: str
	) -> numpy.ndarray:
		if mode == "easy":
			z = numpy.maximum(-numpy.exp(-1), beta * 0.5 + mu)
			sig = numpy.exp(-lambertw(z).real)
		else:
			assert mode == "hard"
			z = numpy.minimum(numpy.exp(-1), beta * 0.5 - mu)
			sig = numpy.exp(-lambertw(-z).real)

		return sig


def compute_flexw(difficulty: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
	# https://doi.org/10.1109/TNNLS.2023.3284430
	# https://openreview.net/forum?id=pSbqyZRKzbw
	r"""
	# alpha x gamma
	easy:		{0.1, 0.2, 0.3}	x	{-1.0, -0.5, -0.4, -0.2}
	hard:		{0.1, 0.2, 0.3}	x	{0.2, 0.4, 0.5, 1.0}
	medium:		{0.4, 0.6, 0.8}	x	{0.2, 0.4, 0.5, 1.0}
	two-ends:	{0.4, 0.6, 0.8}	x	{-1.0, -0.5, -0.4, -0.2}
	"""

	t = difficulty + alpha
	w = torch.pow(t, gamma) * torch.exp(-gamma * t)
	return w


def _compute_rgd_t(loss: torch.Tensor, norm_const: float = 1.0) -> torch.Tensor:
	# Hard-First
	assert norm_const > 0
	weight = 1.0 - torch.clamp(loss.detach(), min=0, max=norm_const) / (norm_const + 1)
	return weight ** (-1)


def _compute_rgd_exp(loss: torch.Tensor, norm_const: float = 1.0) -> torch.Tensor:
	# Hard-First
	assert norm_const > 0
	weight = torch.exp(
		torch.clamp(loss.detach(), min=0, max=norm_const) / (norm_const + 1)
	)
	return weight


def _compute_rgd_inv_t(loss: torch.Tensor, norm_const: float = 1.0) -> torch.Tensor:
	# Easy-First
	assert norm_const > 0
	weight = 1.0 - torch.clamp(loss.detach(), min=0, max=norm_const) / (norm_const + 1)
	return weight


def _compute_rgd_inv_exp(loss: torch.Tensor, norm_const: float = 1.0) -> torch.Tensor:
	# Easy-First
	assert norm_const > 0
	weight = torch.exp(
		-torch.clamp(loss.detach(), min=0, max=norm_const) / (norm_const + 1)
	)
	return weight


class RGD(nn.Module):
	def __init__(self, rgd_mode: str, norm_const: float = 1.0) -> None:
		super(RGD, self).__init__()

		self.rgd_mode = str(rgd_mode)
		self.norm_const = float(norm_const)
		assert self.rgd_mode in {"exp", "t", "inv_exp", "inv_t"} and self.norm_const > 0

		self.fn = {
			"invexp": _compute_rgd_inv_exp,
			"invt": _compute_rgd_inv_t,
			"exp": _compute_rgd_exp,
			"t": _compute_rgd_t,
		}[self.rgd_mode]

	@torch.no_grad()
	def forward(self, loss: torch.Tensor) -> torch.Tensor:
		return self.fn(loss, self.norm_const)
