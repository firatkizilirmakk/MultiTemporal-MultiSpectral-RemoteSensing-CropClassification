import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function that applies Euclidean distance between pair vectors
    """
    def __init__(self, margin = 2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclideanDistance = F.pairwise_distance(output1, output2, keepdim = True)
        loss = torch.mean((label) * torch.pow(euclideanDistance, 2) +
                                      (1 -label) * torch.pow(torch.clamp(self.margin - euclideanDistance, min=0.0), 2))
        return loss
