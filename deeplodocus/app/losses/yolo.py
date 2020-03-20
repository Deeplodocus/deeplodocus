import torch
import torch.nn as nn
from torch.nn.modules import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss


class ObjectLoss(nn.Module):

    def __init__(self, iou_threshold=0.5, obj_weight=0.5):
        super(ObjectLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.bce_loss = BCEWithLogitsLoss(pos_weight=torch.tensor(obj_weight, dtype=torch.float))

    def forward(self, outputs, targets):
        # Unpack YOLO outputs
        predictions, anchors = outputs["detections"], outputs["scaled_anchors"]
        # Unpack some shapes (batch size, n targets, n scales, n anchors
        b, t = targets.shape[0:2]
        s, a = anchors.shape[0:2]
        # Make a mask for labels to ignore
        mask = torch.all(targets[..., 0:4] != -1, dim=2).view(-1)
        # Make tensor of prediction shapes
        shapes = torch.tensor([p.shape[2:4] for p in predictions], device=anchors.device)
        # Get cell indices of each target for each scale (b x t x s x 2)
        target_cells = get_target_cells(targets, shapes)
        # Zero the targets for comparison with prior bounding boxes
        zeroed_targets = self.get_zeroed_targets(targets[..., 0:4], shapes)
        # Calculate Jaccard Index between each zeroed target and each prior bounding box
        overlap = calc_overlap(anchors, zeroed_targets)
        # Suppress or ignore some target-anchors
        gt_values = suppress_anchors(overlap, iou_threshold=self.iou_threshold)
        # Initialise list of ground truth tensors
        ground_truth = [torch.zeros((b, a, h, w), device=anchors.device) for h, w in shapes]
        # Make array of anchor indices
        anchor_index = torch.arange(a).view(1, -1).repeat(torch.sum(mask).item(), 1)
        for scale, values in enumerate(gt_values.view(b * t, s, a).permute(1, 0, 2)):
            cells = target_cells[:, :, scale, :].view(-1, a)[mask].permute(1, 0).view(a, -1, 1)
            values = values[mask]
            # Put ground truth values into ground truth array
            ground_truth[scale].index_put_((cells[0], anchor_index, cells[1], cells[2]), values)
        # Concatenate all ground truths and predictions
        ground_truth = torch.cat([gt.view(-1) for gt in ground_truth], dim=0)
        predictions = torch.cat([p[..., 4].view(-1) for p in predictions], dim=0)
        return self.bce_loss(predictions, ground_truth)

    #def mse_loss(self, prediction, gt):
    #    return self.noobj_weight * torch.mean((prediction[gt == 0] - gt[gt == 0]) ** 2) \
    #           + torch.mean((prediction[gt == 1] - gt[gt == 1]) ** 2)

    @staticmethod
    def get_zeroed_targets(targets, shapes):
        (b, t), s = targets.shape[0:2], shapes.shape[0]
        zeroed_targets = targets.view(b, t, 1, 4) * shapes[:, [1, 0, 1, 0]].view(1, 1, s, 4).float()
        zeroed_targets[..., 0:2] -= (torch.floor(zeroed_targets[..., 0:2]) + 0.5)
        return zeroed_targets


class BoxLoss(nn.Module):

    def __init__(self, iou_threshold=0.5):
        super(BoxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.mse = MSELoss()

    def forward(self, outputs, targets):
        # Unpack YOLO outputs
        predictions, anchors = outputs["detections"], outputs["scaled_anchors"]
        # Unpack some shapes (batch size, n targets, n scales, n anchors
        b, t = targets.shape[0:2]
        s, a = anchors.shape[0:2]
        b = targets.shape[0]
        # Make a mask for labels to ignore
        mask = torch.all(targets[..., 0:4] != -1, dim=2).view(-1)                                       # (b * t)
        # Make tensor of prediction shapes
        shapes = torch.tensor([p.shape[2:4] for p in predictions], device=anchors.device)               # (s, 2)
        # Get cell indices of each target for each scale (b x t x s x 3)
        target_cells = get_target_cells(targets, shapes).view(-1, s, 3)[mask].permute(1, 0, 2)          # (s, ?, 3)
        # Get scaled targets
        scaled_targets = self.get_scaled_targets(targets[..., 0:4], shapes)                             # (b, t, s, 4)
        # Zero the targets for comparison with prior bounding boxes
        zeroed_targets = self.get_zeroed_targets(scaled_targets.clone())
        # Calculate Jaccard Index between each zeroed target and each prior bounding box
        overlap = calc_overlap(anchors, zeroed_targets)
        # Suppress or ignore some target-anchors
        anchor_indices = suppress_anchors(
            overlap,
            iou_threshold=self.iou_threshold
        ).view(-1, s, a)[mask].permute(1, 0, 2)                                                         # (s, ?, 3)
        # Initialise list of ground truth tensors
        ground_truth = [torch.zeros((b, a, h, w, 4), device=anchors.device) for h, w in shapes]
        # Reshape the scaled targets into something more convenient
        scaled_targets = scaled_targets.view(-1, s, 4)[mask].permute(1, 0, 2)                           # (s, ?, 4)
        for scale, anchor_index, cells, coords in zip(range(s), anchor_indices, target_cells, scaled_targets):
            # Remove cells, coords and anchor_index where no anchor is responsible
            cells = cells[~torch.all(anchor_index == 0, dim=1)].t()
            coords = coords[~torch.all(anchor_index == 0, dim=1)]
            anchor_index = anchor_index[~torch.all(anchor_index == 0, dim=1)]
            if anchor_index.shape[0]:
                # Get the anchor index (as a number)
                _, anchor_index = torch.max(anchor_index, dim=1)
                # Put ground truth values into ground truth array
                ground_truth[scale].index_put_((cells[0], anchor_index, cells[1], cells[2]), coords)
        ground_truth = torch.cat([gt.view(-1, 4) for gt in ground_truth], dim=0)
        predictions = torch.cat([pred[..., 0:4].view(-1, 4) for pred in predictions], dim=0)
        predictions = predictions[~ torch.all(ground_truth == 0, dim=1)]
        ground_truth = ground_truth[~ torch.all(ground_truth == 0, dim=1)]
        return self.mse(
            ground_truth[:, 0:2].contiguous().view(-1),
            predictions[:, 0:2].contiguous().view(-1)
        ) \
            + self.mse(
                torch.sqrt(ground_truth[:, 2:4].contiguous().view(-1)),
                torch.sqrt(predictions[:, 2:4].contiguous().view(-1))
        )

    @staticmethod
    def get_scaled_targets(targets, shapes):
        (b, t), s = targets.shape[0:2], shapes.shape[0]
        return targets.view(b, t, 1, 4) * shapes[:, [1, 0, 1, 0]].view(1, 1, s, 4).float()

    @staticmethod
    def get_zeroed_targets(targets):
        targets[..., 0:2] -= (torch.floor(targets[..., 0:2]) + 0.5)
        return targets


class ClassLoss(nn.Module):

    def __init__(self, weights=None):
        super(ClassLoss, self).__init__()
        None if weights is None else torch.tensor(weights)
        self.cross_entropy = CrossEntropyLoss(ignore_index=-1, weight=weights)

    def forward(self, outputs, targets):
        # Unpack YOLO outputs
        predictions, anchors = outputs["detections"], outputs["scaled_anchors"]
        # Unpack some shapes (batch size, n targets, n scales, n anchors
        b, t = targets.shape[0:2]
        s, a = anchors.shape[0:2]
        n_cls = predictions[0].shape[-1] - 5
        # Make a mask for labels to ignore
        mask = torch.all(targets[..., 0:4] != -1, dim=2).view(-1)
        # Make tensor of prediction shapes
        shapes = torch.tensor([p.shape[2:4] for p in predictions], device=anchors.device)
        # Get cell indices of each target for each scale (b x t x s x 2)
        target_cells = get_target_cells(targets, shapes)
        # Initialise list of ground truth tensors
        ground_truth = [torch.zeros((b, a, h, w), device=anchors.device).fill_(-1) for h, w in shapes]
        # Make array of anchor indices
        anchor_index = torch.arange(a).view(1, -1).repeat(torch.sum(mask).item(), 1)
        for scale in range(s):
            # Shape the target_cells into something more convenient
            cells = target_cells[:, :, scale, :].view(-1, a, 1)[mask].permute(1, 0, 2)
            # Put target class values into ground truth array
            ground_truth[scale].index_put_(
                (cells[0], anchor_index, cells[1], cells[2]),
                targets[..., 4].view(-1, 1)[mask].repeat(1, a)
            )
        # Concatenate all ground truths and predictions
        ground_truth = torch.cat([gt.view(-1) for gt in ground_truth], dim=0)
        predictions = torch.cat([p[..., 5:].view(-1, n_cls) for p in predictions], dim=0)
        if ground_truth.shape[0]:
            return self.cross_entropy(predictions, ground_truth.long())
        else:
            return torch.tensor(0, dtype=torch.float32, device=anchors.device, requires_grad=True)


def calc_overlap(anchors, targets):
    (s, a), (b, t) = anchors.shape[0:2], targets.shape[0:2]
    anchors = anchors.view(1, 1, s, a, 2)
    targets = targets.view(b, t, s, 1, 4)
    lower_bound = torch.max(-anchors / 2, targets[..., 0:2] - targets[..., 2:4] / 2)
    upper_bound = torch.min(anchors / 2, targets[..., 0:2] + targets[..., 2:4] / 2)
    mask = torch.prod((lower_bound < upper_bound).float(), dim=4)
    intersection = torch.prod((upper_bound - lower_bound), dim=4) * mask
    anchor_area = torch.prod(anchors, dim=4)
    target_area = (targets[..., 2] - targets[..., 0]) * (targets[..., 3] - targets[..., 1])
    return intersection / (anchor_area + target_area - intersection)


def get_target_cells(targets, shapes):
    (b, t), s = targets.shape[0:2], shapes.shape[0]
    return torch.cat(
        (
            torch.arange(b, dtype=torch.long, device=targets.device).view(b, 1, 1, 1).repeat(1, t, s, 1),
            (targets[..., [1, 0]].view(b, t, 1, 2) * shapes.view(1, 1, s, 2).float()).long()
        ), dim=3
    )


def suppress_anchors(overlap, iou_threshold=0.5):
    b, t, s, a = overlap.shape
    overlap = overlap.view(b * t, s * a)
    overlap[overlap == torch.max(overlap, dim=1)[0].view(-1, 1)] = 1
    overlap[overlap < iou_threshold] = 0
    overlap[(overlap != 0) & (overlap != 1)] = -1
    return overlap.view(b, t, s, a)