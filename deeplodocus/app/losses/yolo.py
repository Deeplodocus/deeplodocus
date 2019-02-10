import torch
import torch.nn as nn
from torch.nn.modules import BCELoss, CrossEntropyLoss, MSELoss


class ObjectLoss(nn.Module):

    def __init__(self, threshold=0.5, noobj_weight=0.5, obj_weight=1):
        super(ObjectLoss, self).__init__()
        self.threshold = threshold
        self.threshold = threshold
        self.noobj_weight = noobj_weight
        self.obj_weight = obj_weight
        self.bce = BCELoss()

    def forward(self, outputs, targets):
        loss = torch.tensor(0.0)
        for output, anchors in outputs:                                     # For outputs and anchors from each layer
            batch_size, num_anchors, h, w, _ = output.shape
            for b in range(batch_size):
                # Targets from current batch and remove targets with all zeros
                target = targets[b, :, 1:5]
                target = target[~torch.all(target == 0, dim=1)]

                # Objectness predictions for current batch (h x w x a)
                prediction = output[b, :, :, :, 4].view(-1)

                # Get x and y coords of 'responsible' cells
                cx, cy = responsible_cell(target[:, 0:2], (h, w))

                # Rescale target by (w, h) and move (x, y) to be relative to anchors
                target = center_target(target, (cy, cx), (h, w), prediction.is_cuda)

                # Calculate the iou between targets and anchors
                ious = iou(target, anchors)

                # Indexes of 'responsible' anchors and create a mask for thresholding by iou
                mask, anchor_index = torch.max(ious, 1)
                mask = mask > self.threshold

                # Init the truth values as one hot encoded 'responsible' anchors
                truth = torch.eye(num_anchors)[anchor_index]
                if prediction.is_cuda:
                    truth = truth.cuda()

                # Mask values where iou does not exceed threshold and set zeros to ignore index
                truth *= mask.type(torch.float)
                truth[truth == 0] = -100

                # Init the ground truth with 0
                ground_truth = torch.zeros((h, w))
                if prediction.is_cuda:
                    ground_truth = ground_truth.cuda()

                # Repeat ground truth for each anchor. include true values for 'responsible cells' and reshape
                ground_truth = ground_truth.view(*ground_truth.shape, 1).repeat(1, 1, num_anchors)
                ground_truth = ground_truth.index_put_(tuple(torch.stack((cy, cx), 1).t().type(torch.long)), truth)
                ground_truth = ground_truth.view(-1)

                # Set weights for empty cells and occupied cells
                weight_obj = (ground_truth == 1).type(torch.float) * self.obj_weight
                weight_noobj = (ground_truth == 0).type(torch.float) * self.noobj_weight
                weight = weight_obj + weight_noobj

                if prediction.is_cuda:
                    loss = loss.cuda()
                    weight = weight.cuda()

                self.bce = BCELoss(weight=weight)
                loss += self.bce(prediction, ground_truth)
        return loss / batch_size


class ClassLoss(nn.Module):

    def __init__(self):
        super(ClassLoss, self).__init__()
        self.cross_entropy = CrossEntropyLoss(ignore_index=-100)

    def forward(self, outputs, targets):
        loss = torch.tensor(0.0)
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
                if prediction.is_cuda:
                    ground_truth = ground_truth.cuda()
                    loss = loss.cuda()

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
        loss = torch.tensor(0.0)
        for output, anchors in outputs:
            batch_size, num_anchors, h, w, _ = output.shape
            for b in range(batch_size):
                # Targets from current batch and remove targets with all zeros
                target = targets[b, :, 1:5]
                target = target[~torch.all(target == 0, dim=1)]

                # Box predictions for current batch (x, y, w, h)
                prediction = output[b, :, :, :, 0:4].view(-1, 4)

                # Get x and y coords of 'responsible' cells
                cx, cy = responsible_cell(target[:, 0:2], (h, w))

                # rel_target = target centred at (0, 0) for calculating iou with anchors
                centred_target = center_target(target.clone(), (cy, cx), (h, w), prediction.is_cuda)

                # Scale target position by w and h in line with prediction
                target = scale(target, (h, w))

                # Calculate the iou between targets and anchors
                ious = iou(centred_target, anchors)

                # Indexes of 'responsible' anchors and create a mask for thresholding by iou
                mask, anchor_index = torch.max(ious, 1)
                anchor_index = torch.eye(num_anchors)[anchor_index]
                if prediction.is_cuda:
                    mask = mask.cuda()
                    anchor_index = anchor_index.cuda()
                mask = mask > self.threshold
                mask = mask.type(torch.float) * anchor_index

                # Mask targets that are not required
                target = target.view(-1, 1, 4) * mask.view(-1, num_anchors, 1)

                # Init the ground truth with ignore index (-100)
                gt = torch.zeros((h, w, num_anchors, 4))
                if prediction.is_cuda:
                    gt = gt.cuda()
                    loss = loss.cuda()

                # Put true values into ground_truth, repeat for each anchor and reshape
                cells = torch.stack((cy, cx), 1).type(torch.long)
                gt = gt.index_put_(tuple(cells.t()), target)
                gt = gt.view(-1, 4)

                # Stack targets and predictions and remove rows where all targets are zero
                gt = torch.stack((prediction, gt), 1)
                gt = gt[~torch.all(gt[:, 1, :] == 0, dim=1)]

                loss_ = (self.mse(gt[:, 0, 0:2], gt[:, 1, 0:2])
                         + self.mse(torch.rsqrt(gt[:, 0, 2:4]), torch.rsqrt(gt[:, 1, 2:4])))
                if not isnan(loss_):
                    loss += loss_
        return loss / batch_size


def center_target(target, cell, shape, cuda=False):
    """
    target centred at (0, 0) for calculating iou with anchors
    :param target:
    :param cell:
    :param shape:
    :param cuda:
    :return:
    """
    cy, cx = cell
    h, w = shape
    target[:, 0] = (target[:, 0] - (cx + 0.5) / w) * w
    target[:, 1] = (target[:, 1] - (cy + 0.5) / h) * h
    target[:, 2] *= w
    target[:, 3] *= h
    if cuda:
        return target.cuda()
    else:
        return target


def iou(targets, anchors):
    """
    :param targets:
    :param anchors:
    :return:
    """
    tx0 = targets[:, 0] - targets[:, 2] / 2
    ty0 = targets[:, 1] - targets[:, 3] / 2
    tx1 = targets[:, 0] + targets[:, 2] / 2
    ty1 = targets[:, 1] + targets[:, 3] / 2
    ax0 = - anchors[:, 0] / 2
    ay0 = - anchors[:, 1] / 2
    ax1 = anchors[:, 0] / 2
    ay1 = anchors[:, 1] / 2
    x0 = torch.max(tx0.view(-1, 1), ax0.view(1, -1))
    y0 = torch.max(ty0.view(-1, 1), ay0.view(1, -1))
    x1 = torch.min(tx1.view(-1, 1), ax1.view(1, -1))
    y1 = torch.min(ty1.view(-1, 1), ay1.view(1, -1))
    flag = ((x0 < x1) * (y0 < y1)).type(torch.float)
    intersection = ((x1 - x0) * (y1 - y0)) * flag
    a_area = torch.prod(anchors, 1)
    t_area = torch.prod(targets[:, 2:4], 1)
    combined_area = t_area.view(-1, 1) + a_area.view(1, -1)
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


def responsible_cell(target, shape):
    """
    :param target:
    :param shape:
    :return:
    """
    h, w = shape
    return torch.floor(target[:, 0] * w), torch.floor(target[:, 1] * h)


def isnan(x):
    """
    :param x:
    :return:
    """
    return x != x
