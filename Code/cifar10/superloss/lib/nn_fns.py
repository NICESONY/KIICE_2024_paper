import numpy
import torch
from typing import Any


def train_superloss_one_epoch(
	network: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
	superloss: torch.nn.Module,
	epoch: int,
	device: torch.device,
	**kwargs: Any,
) -> None:
	network.train()

	for batch_idx, (x, y) in enumerate(loader):
		assert x.ndim == 4 and y.ndim == 1

		x = x.to(device)
		y = y.to(device)

		optimizer.zero_grad()
		logit = network(x)
		loss_orig = criterion(logit, y)
		loss, sig = superloss(loss_orig)
		loss = loss.mean()
		loss.backward()

		optimizer.step()

		print(
			f"\rEpoch {epoch:3d} {numpy.float32(batch_idx+1) / numpy.float32(len(loader)) * 100:3.2f} loss {loss_orig.mean().tolist():.4f} sig {sig.mean().tolist():.4f} tau {superloss.tau:.4f}",
			end="",
		)
