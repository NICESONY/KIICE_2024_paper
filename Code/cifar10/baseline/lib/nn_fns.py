import numpy
import torch
from typing import Any, Tuple


def train_one_epoch(
	network: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: torch.nn.Module,
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
		loss = criterion(logit, y)
		loss.backward()

		optimizer.step()

		print(
			f"\rEpoch {epoch:3d} {numpy.float32(batch_idx+1) / numpy.float32(len(loader)) * 100:3.2f} loss {loss.tolist():.4f}",
			end="",
		)


def inference(
	network: torch.nn.Module,
	loader: torch.utils.data.DataLoader,
	device: torch.device,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
	network.eval()

	pred = []
	target = []

	with torch.no_grad():
		for x, y in loader:
			assert x.ndim == 4 and y.ndim == 1
			x = x.to(device)
			# y = y.to(device)
			logit = network(x)

			pred.append(logit.cpu().numpy())
			target.append(y.numpy())

	pred = numpy.vstack(pred).astype(numpy.int32)
	target = numpy.hstack(target).astype(numpy.int32)

	assert pred.shape[0] == target.shape[0]

	return pred, target
