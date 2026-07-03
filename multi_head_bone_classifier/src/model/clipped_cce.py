from torch import nn
import torch.nn.functional as F


class ClippedCrossEntropyLoss(nn.Module):
    def __init__(self, clip_p=0.7, weight=None):
        super(ClippedCrossEntropyLoss, self).__init__()
        self.clip_p = clip_p
        self.weight = weight

    def forward(self, logits, target):
        # Take crossentroy loss
        losses = F.cross_entropy(logits, target, reduction="none", weight=self.weight)
        # Take softmax of logits
        probs = F.softmax(logits, dim=-1)
        # For each batch take the prob of the target class
        probs = probs[range(target.shape[0]), target]
        # Make a masking tensor where each element is 1 if the prob is less than the clip_p
        # else it is a zero
        mask = (probs <= self.clip_p).float()
        # Multiply the mask with the losses
        losses = mask * losses
        loss = losses.mean()
        return loss
