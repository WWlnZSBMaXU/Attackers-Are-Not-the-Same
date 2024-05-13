import torch
import sys
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class KlDivLoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        logits: bool = True,
    ) -> None:
        super(KlDivLoss, self).__init__(size_average, reduce, reduction)
        self.logits = logits
        self.eps = 1e-12

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if self.logits:
            pred = torch.softmax(pred, dim=-1)

        loss = (-target * torch.log((pred / (target + self.eps)) + self.eps)).sum(
            dim=-1
        )

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction is None:
            return loss
        else:
            sys.exit(f"Invalid reduction type: {self.reduction}")


class SoftCrossEntropyLoss(_Loss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        logits: bool = True,
    ) -> None:
        super(SoftCrossEntropyLoss, self).__init__(size_average, reduce, reduction)
        self.logits = logits
        self.eps = 1e-12

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if self.logits:
            loss = (-target * torch.log_softmax(pred, dim=-1)).sum(dim=-1)
        else:
            loss = (-target * torch.log(pred + self.eps)).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction is None:
            return loss
        else:
            sys.exit(f"Invalid reduction type: {self.reduction}")


class EntropyLoss(_Loss):
    def __init__(self, logits=True, reduction="mean"):
        super(EntropyLoss, self).__init__()
        self.logits = logits
        self.eps = 1e-12
        self.reduction = reduction

    def forward(self, x):
        if self.logits:
            loss = -(F.softmax(x, dim=1) * F.log_softmax(x, dim=1)).sum(dim=-1)
        else:
            loss = -(x * torch.log(x + self.eps)).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction is None:
            return loss
        else:
            sys.exit(f"Invalid reduction type: {self.reduction}")