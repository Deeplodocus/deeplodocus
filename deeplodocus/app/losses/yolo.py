import torch
import torch.nn as nn
from torch.nn.modules import BCELoss, CrossEntropyLoss, MSELoss


class ObjectLoss(nn.Module):

    def __init__(self, threshold=0.5):
        super(ObjectLoss, self).__init__()
        self.threshold = threshold
        self.bce = BCELoss()

    def forward(self, outputs, targets):
        loss = torch.tensor(0.0)
        for output, anchors in outputs:                             # For outputs and anchors from each layer
            prediction = output[..., 4]                             # Get objectness predictions
            ground_truth = torch.zeros_like(prediction)             # Initialise ground truth objectness with zeros
            batch_size, num_anchors, h, w = prediction.shape        # Get b, a, h, w from prediction shape
            if output.is_cuda:
                ground_truth = ground_truth.cuda()
                loss = loss.cuda()
            for b in range(batch_size):                             # For each batch
                for target in targets[b, :, 1:]:                    # targets is b x t x 5 (t = number of targets)
                    t_a, t_j, t_i, iou = find_cell(anchors, target, (h, w))  # Get the a, h, w indices and corresponding iou
                    if iou > self.threshold:                        # If the anchor overlaps the target
                        ground_truth[b, t_a, t_j, t_i] = 1          # Set the ground truth to 1 (object is present)
            prediction = prediction.view(-1)                        # Flatten prediction
            ground_truth = ground_truth.view(-1)                    # Flatten ground truth
            loss += self.bce(prediction, ground_truth)              # Calculate binary cross entropy loss
        return loss


class ClassLoss(nn.Module):

    def __init__(self, threshold=0.5):
        super(ClassLoss, self).__init__()
        self.threshold = threshold
        self.cross_entropy = CrossEntropyLoss()

    def forward(self, outputs, targets):
        loss = torch.tensor(0.0)                                    # Stores the loss for each layer
        for output, anchors in outputs:                             # For outputs and anchors from each layer
            prediction = output[..., 5:]                            # Get class predictions
            batch_size, num_anchors, h, w, num_classes = prediction.shape
            ground_truth = torch.zeros(0, dtype=torch.long)
            active_prediction = torch.zeros((0, num_classes), dtype=prediction.dtype)
            if output.is_cuda:
                loss = loss.cuda()
                ground_truth = ground_truth.cuda()
                active_prediction = active_prediction.cuda()
            for b in range(batch_size):                             # For each batch
                for target in targets[b, ...]:                      # targets is b x t x 5 (t = number of targets)
                    t_a, t_j, t_i, iou = find_cell(anchors, target[1:], (h, w))     # Cell indices and iou
                    if iou > self.threshold:                                        # If the anchor overlaps the target
                        ground_truth = torch.cat((ground_truth, target[0].view(-1).type(torch.long)))
                        active_prediction = torch.cat(
                            (active_prediction, prediction[b, t_a, t_j, t_i, :].view(1, -1))
                        )
            if active_prediction.size(0):
                loss += self.cross_entropy(active_prediction, ground_truth)  # Calculate cross entropy loss
        return loss


class BoxLoss(nn.Module):

    def __init__(self, threshold=-0.5):
        super(BoxLoss, self).__init__()
        self.threshold = threshold
        self.mse = MSELoss()

    def forward(self, outputs, targets):
        loss = torch.tensor(0.0)
        for output, anchors in outputs:                                 # For outputs and anchors from each layer
            prediction = output[..., :4]                                # Get box predictions
            batch_size, num_anchors, h, w, _ = prediction.shape
            ground_truth = torch.zeros((0, 4), dtype=prediction.dtype)
            active_prediction = torch.zeros((0, 4), dtype=prediction.dtype)
            if output.is_cuda:
                ground_truth = ground_truth.cuda()
                active_prediction = active_prediction.cuda()
                loss = loss.cuda()
            for b in range(batch_size):                                 # For each batch
                for target in targets[b, :, 1:]:                        # targets is b x t x 5 (t = number of targets)
                    t_a, t_j, t_i, iou = find_cell(anchors, target, (h, w))     # Cell indices and corresponding iou
                    if iou > self.threshold:                                    # If the anchor overlaps the target
                        ground_truth = torch.cat((ground_truth, target.view(1, -1)))
                        active_prediction = torch.cat(
                            (active_prediction, prediction[b, t_a, t_j, t_i, :].view(1, -1))
                        )
            if active_prediction.size(0):
                loss += self.mse(active_prediction, ground_truth)
        return loss


def find_cell(anchors, target, shape):
    h, w = shape
    t_i = int(target[0] * w)            # Cell width index
    t_j = int(target[1] * h)            # Cell height index
    t_w = target[2] * w                 # Scaled width of the target box
    t_h = target[3] * h                 # Scaled height of the target box
    t_a, iou = match_anchors(torch.tensor((t_w, t_h)), anchors)     # Index and iou of the best matching anchor
    return t_a, t_j, t_i, iou


def match_anchors(target, anchors):
    intersections = torch.clamp(anchors[:, 0], max=target[0]) * torch.clamp(anchors[:, 1], max=target[1])
    combined_area = torch.prod(anchors, 1) + torch.prod(target)
    ious = intersections / (combined_area - intersections)
    return torch.argmax(ious), torch.max(ious)
