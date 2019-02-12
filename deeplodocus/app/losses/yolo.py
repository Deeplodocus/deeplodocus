import torch
import torch.nn as nn
from torch.nn.modules import BCELoss, CrossEntropyLoss, MSELoss


class ObjectLoss(nn.Module):

    def __init__(self, threshold=0.5, noobj_weight=0.5, obj_weight=1):
        super(ObjectLoss, self).__init__()
        self.threshold = threshold
        self.noobj_weight = noobj_weight
        self.obj_weight = obj_weight
        self.bce = BCELoss()

    def forward(self, outputs, targets):
        # Find out we are using a GPU
        is_cuda = targets.is_cuda

        # Initialise the loss tensor and make a tensor of 0.5
        loss = torch.tensor(0.0, requires_grad=True)
        half = torch.tensor(0.5)
        if is_cuda:
            loss = loss.cuda()
            half = half.cuda()

        for output, anchors in outputs:                                     # For outputs and anchors from each layer
            batch_size, num_anchors, h, w, _ = output.shape
            anchors = anchors[0:num_anchors, :]

            # Make a tensor of zeros (a x 2) to stack onto the anchors (for IOU)
            zeros = torch.zeros((num_anchors, 2))
            if is_cuda:
                zeros = zeros.cuda()

            for b in range(batch_size):
                target = targets[b, :, 1:5]
                target = target[~torch.all(target == 0, dim=1)]
                scale(target, (h, w))

                # Get the cell indices for each target and create comparable anchors and targets about (0, 0)
                cx, cy = torch.floor(target[:, 0:2]).t()
                zero_anchors = torch.cat((zeros, anchors), 1)
                zero_target = target.clone()
                zero_target[:, 0:2] -= (torch.stack((cx, cy), 1) + half)

                # Calculate the IOU of targets and anchors and the index of the best anchor
                overlap, anchor = torch.max(iou(xywh2rect(zero_target), xywh2rect(zero_anchors)), 1)
                mask = (overlap > self.threshold)

                # Tensor of indices of responsible cells (y, x)
                gt_index = torch.stack((cy, cx), 1).type(torch.long)
                gt_index = gt_index[mask]
                anchor = anchor[mask]
                anchor = torch.eye(num_anchors)[anchor]

                # Put the target into the corresponding index in the ground truth tensor, reshape
                gt = torch.zeros((h, w, num_anchors)).type(torch.float)
                if is_cuda:
                    anchor = anchor.cuda()
                    gt = gt.cuda()
                gt = gt.index_put_(tuple(gt_index.t()), anchor)
                gt = gt.permute(2, 0, 1)
                gt = gt.contiguous().view(-1)

                # Objectness predictions for current batch (a x h x w)
                prediction = output[b, :, :, :, 4].view(-1)

                # Set weights for empty cells and occupied cells
                weight_obj = (gt == 1).type(torch.float) * self.obj_weight
                weight_noobj = (gt == 0).type(torch.float) * self.noobj_weight
                weight = weight_obj + weight_noobj

                loss += BCELoss(weight=weight)(prediction, gt)

        return loss / batch_size


class ClassLoss(nn.Module):

    def __init__(self):
        super(ClassLoss, self).__init__()
        self.cross_entropy = CrossEntropyLoss(ignore_index=-100)

    def forward(self, outputs, targets):
        # Find out we are using a GPU
        is_cuda = targets.is_cuda

        # Initialise the loss tensor
        loss = torch.tensor(0.0, requires_grad=True)
        if is_cuda:
            loss = loss.cuda()

        # For the output of each layer and the corresponding anchors
        for output, _ in outputs:
            batch_size, num_anchors, h, w, num_classes = output[..., 5:].shape

            for b in range(batch_size):
                # Targets from current batch and remove targets with all zeros
                target = targets[b, ...]
                target = target[~torch.all(target == 0, dim=1)]

                # Class predictions for current batch (c0, c1, c2, ...)
                prediction = output[b, :, :, :, 5:].view(-1, num_classes)

                # Get the indices of 'responsible' cells
                cells = torch.stack((target[:, 2] * h, target[:, 1] * w), 1).type(torch.long)

                # Init the ground truth with ignore index (-100)
                ground_truth = torch.zeros((h, w)).fill_(-100)
                if is_cuda:
                    ground_truth = ground_truth.cuda()

                # Put true values into ground_truth, repeat for each anchor and reshape
                ground_truth = ground_truth.index_put_(tuple(cells.t()), target[:, 0]).type(torch.long)
                ground_truth = ground_truth.view(*ground_truth.shape, 1).repeat(1, 1, num_anchors)
                ground_truth = ground_truth.view(-1)

                # Calculate and accumulate loss (inside batch loop so divide by batch size)
                loss += (self.cross_entropy(prediction, ground_truth))
        return loss / batch_size


class BoxLoss(nn.Module):

    def __init__(self, threshold=0.5):
        super(BoxLoss, self).__init__()
        self.threshold = threshold
        self.mse = MSELoss()

    def forward(self, outputs, targets):
        # Find out we are using a GPU
        is_cuda = targets.is_cuda

        # Initialise the loss tensor and make a tensor of 0.5
        loss = torch.tensor(0.0, requires_grad=True)
        half = torch.tensor(0.5)
        if is_cuda:
            loss = loss.cuda()
            half = half.cuda()

        # For the output of each layer and the corresponding anchors
        for output, anchors in outputs:
            batch_size, num_anchors, h, w, _ = output.shape
            anchors = anchors[0:num_anchors, :]

            # Make a tensor of zeros (a x 2) to stack onto the anchors (for IOU)
            zeros = torch.zeros((num_anchors, 2))
            if is_cuda:
                zeros = zeros.cuda()

            for b in range(batch_size):
                # Get the target for this batch, remove rows of all zero and scale by (h, w)
                target = targets[b, :, 1:5]
                target = target[~torch.all(target == 0, dim=1)]
                scale(target, (h, w))

                # Get the cell indices for each target and create comparable anchors and targets about (0, 0)
                cx, cy = torch.floor(target[:, 0:2]).t()
                zero_anchors = torch.cat((zeros, anchors), 1)
                zero_target = target.clone()
                zero_target[:, 0:2] -= (torch.stack((cx, cy), 1) + half)

                # Calculate the IOU of targets and anchors and the index of the best anchor
                overlap, anchor = torch.max(iou(xywh2rect(zero_target), xywh2rect(zero_anchors)), 1)
                mask = (overlap > self.threshold)

                # Tensor of indices of responsible cells (a, y, x)
                gt_index = torch.stack((anchor.type(torch.float), cy, cx), 1).type(torch.long)
                gt_index = gt_index[mask]

                # Put the target into the corresponding index in the ground truth tensor, reshape
                gt = torch.zeros((num_anchors, h, w, 4))
                if is_cuda:
                    gt = gt.cuda()
                gt = gt.index_put_(tuple(gt_index.t()), target).view(-1, 4)

                # Get predictions for current batch and reshape
                prediction = output[b, :, :, :, 0:4].view(-1, 4)

                # STACK PREDICTIONS ONTO GROUND TRUTH TENSOR AND REMOVE ROWS WHERE TARGET == [0, 0, 0, 0]
                gt = torch.stack((prediction, gt), 1)
                gt = gt[~torch.all(gt[:, 1, :] == 0, dim=1)]

                if gt.size(0):
                    loss += self.mse(gt[:, 0, 0:2], gt[:, 1, 0:2])
                    loss += self.mse(torch.rsqrt(gt[:, 0, 2:4]), torch.rsqrt(gt[:, 1, 2:4]))
        return loss / batch_size


def xywh2rect(xywh):
    """
    :param xywh:
    :return:
    """
    rect = torch.zeros_like(xywh.t())
    rect[0] = xywh.t()[0] - xywh.t()[2] / 2
    rect[1] = xywh.t()[1] - xywh.t()[3] / 2
    rect[2] = xywh.t()[0] + xywh.t()[2] / 2
    rect[3] = xywh.t()[1] + xywh.t()[3] / 2
    return rect.t()


def area(tensor):
    """
    :param tensor:
    :return:
    """
    return (tensor[:, 2] - tensor[:, 0]) * (tensor[:, 3] - tensor[:, 1])


def iou(targets, anchors):
    """
    :param targets:
    :param anchors:
    :return:
    """
    x0 = torch.max(targets[:, 0].view(-1, 1), anchors[:, 0].view(1, -1))
    y0 = torch.max(targets[:, 1].view(-1, 1), anchors[:, 1].view(1, -1))
    x1 = torch.min(targets[:, 2].view(-1, 1), anchors[:, 2].view(1, -1))
    y1 = torch.min(targets[:, 3].view(-1, 1), anchors[:, 3].view(1, -1))
    flag = ((x0 < x1) * (y0 < y1)).type(torch.float)
    intersection = ((x1 - x0) * (y1 - y0)) * flag
    combined_area = area(targets).view(-1, 1) + area(anchors).view(1, -1)
    return intersection / (combined_area - intersection)


def scale(target, shape):
    """
    :param target:
    :param shape:
    :return:
    """
    h, w = shape
    target[:, 0] *= w
    target[:, 1] *= h
    target[:, 2] *= w
    target[:, 3] *= h
    return target
