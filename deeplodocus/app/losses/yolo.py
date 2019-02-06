import torch
import torch.nn as nn
from torch.nn.modules import BCELoss


class ObjectLoss(nn.Module):

    def __init__(self, threshold=0.5):
        super(ObjectLoss, self).__init__()
        self.threshold = threshold
        self.bce = BCELoss()

    def forward(self, outputs, targets):
        losses = []
        for output, anchors in outputs:                         # For outputs and anchors from each layer
            prediction = output[..., 4]                         # Get objectness predictions
            ground_truth = torch.zeros_like(prediction)         # Initialise ground truth objectness with zeros
            batch_size, num_anchors, h, w = prediction.shape    # Get b, a, h, w from obj shape
            for b in range(batch_size):                         # For each batch
                for target in targets[b, :, 1:]:                # targets is b x t x 5 (t = number of targets)
                    t_i = int(target[0] * w)                    # Get the cell indices
                    t_j = int(target[1] * h)
                    t_w = target[2] * w                         # Get the gt w and height
                    t_h = target[3] * h
                    # Get index and iou of the best matching anchor
                    t_a, iou = self.match_anchors(torch.tensor((t_w, t_h)), anchors)
                    if iou > self.threshold:
                        ground_truth[b, t_a, t_j, t_i] = 1
            prediction = prediction.view(-1)
            ground_truth = ground_truth.view(-1)
            losses.append(self.bce(prediction, ground_truth))
        return sum(losses)

    @staticmethod
    def match_anchors(target, anchors):
        intersections = torch.clamp(anchors[:, 0], max=target[0]) * torch.clamp(anchors[:, 1], max=target[1])
        combined_area = torch.prod(anchors, 1) + torch.prod(target)
        ious = intersections / (combined_area - intersections)
        return torch.argmax(ious), torch.max(ious)




