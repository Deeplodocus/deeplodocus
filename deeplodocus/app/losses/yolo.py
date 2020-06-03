import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import numpy as np
import cv2


class YOLOLoss(nn.Module):

    def __init__(self, iou_threshold=0.5, box_weight=3.54, obj_weight=64.3, cls_weight=37.4, obj_pos_weight=1.0, cls_pos_weight=1.0, giou=False):
        super().__init__()
        self.obj_bce = None
        self.cls_bce = None
        self.iou_threshold = iou_threshold
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.obj_pos_weight = obj_pos_weight
        self.cls_pos_weight = cls_pos_weight
        self.giou = giou

    def forward(self, outputs, targets):
        ns, na, _ = outputs.anchors.shape
        targets = self.compress_targets(targets)
        anchors = self.make_anchors(outputs)
        target_boxes, c = self.make_target_boxes(targets[:, 2:6], outputs.strides)

        obj = self.jaccard_index(
            self.xywh2rect(target_boxes).view(-1, ns, 1, 4),
            self.xywh2rect(anchors).view(1, ns, na, 4)
        ) > self.iou_threshold

        indices = self.make_indices(targets, obj, c, ns)
        p_box, p_obj, p_cls = self.extract_inference(outputs.inference, indices)
        t_box, t_obj, t_cls = self.initialise_gt(outputs.inference, indices)

        for s in range(ns):
            b, ci, cj, a = indices[s]
            # Box
            t_box[s] = target_boxes[:, s].view(-1, 1, 4).repeat(1, na, 1)[obj[:, s]]
            p_xy = torch.sigmoid(p_box[s][b, a, cj, ci, 0:2])
            p_wh = torch.exp(p_box[s][b, a, cj, ci, 2:4]).clamp(max=1e3) * anchors[s, a, 2:4]
            p_box[s] = torch.cat((p_xy, p_wh), dim=1)
            # Obj
            t_obj[s][b, a, cj, ci] = 1
            # Cls
            t = targets[:, 1].long().view(-1, 1).repeat(1, na)[obj[:, s]]
            t_cls[s][range(len(t)), t] = 1

        box_loss = sum([self.box_loss(p, t) for p, t in zip(p_box, t_box)]) * self.box_weight
        obj_loss = sum([self.obj_bce(p, t) for p, t in zip(p_obj, t_obj)]) * self.obj_weight
        cls_loss = sum([self.cls_bce(p, t) for p, t in zip(p_cls, t_cls)]) * self.cls_weight
        return box_loss + obj_loss + cls_loss

    def box_loss(self, p, t):
        iou = self.jaccard_index(self.xywh2rect(p), self.xywh2rect(t), giou=self.giou)
        return torch.mean(1 - iou)

    def jaccard_index(self, a1, a2, giou=False):
        x0 = torch.max(a1[..., 0], a2[..., 0])
        y0 = torch.max(a1[..., 1], a2[..., 1])
        x1 = torch.min(a1[..., 2], a2[..., 2])
        y1 = torch.min(a1[..., 3], a2[..., 3])
        mask = (x0 < x1) * (y0 < y1)
        intersection = (x1 - x0) * (y1 - y0) * mask.type(torch.float32)
        union = self.area(a1) + self.area(a2) - intersection
        if giou:
            cw = torch.max(a1[..., 2], a2[..., 2]) - torch.min(a1[..., 0], a2[..., 0])
            ch = torch.max(a1[..., 3], a2[..., 3]) - torch.min(a1[..., 1], a2[..., 1])
            c_area = cw * ch + 1e-16
            return intersection / union - (c_area - union) / c_area
        else:
            return intersection / union

    @staticmethod
    def compress_targets(targets):
        b, n, _ = targets.shape
        batch_index = torch.arange(b, device=targets.device).repeat_interleave(n)  # (b * n)
        targets = torch.cat(
            (
                batch_index.view(b, n, 1).float(),
                targets[..., 4].view(b, n, 1),
                targets[..., 0:4]),
            dim=2
        ).view(-1, 6)
        return targets[~torch.all(targets[:, 1:6] == -1, dim=1)]

    @staticmethod
    def make_target_boxes(targets, strides):
        target_boxes = targets.view(-1, 1, 4).repeat(1, len(strides), 1) / strides.view(1, 3, 1)
        c = torch.floor(target_boxes[..., 0:2])
        target_boxes[..., 0:2] -= c
        return target_boxes, c.long()

    @staticmethod
    def make_indices(targets, obj, c, s):
        n, _, a = obj.shape  # Number of anchors
        b = targets[:, 0].long()  # Get batch indices
        indices = []
        for scale in range(s):
            ci, cj = c[:, scale].T  # Get cell indices
            i = torch.cat(
                (
                    torch.stack((b, ci, cj), dim=1).view(-1, 1, 3).repeat(1, a, 1),
                    torch.arange(a, device=targets.device).view(1, a, 1).repeat(n, 1, 1)
                ), dim=2).view(-1, 4)
            indices.append(i[obj[:, scale].reshape(-1)].T)
        return indices

    @staticmethod
    def initialise_gt(inference, indices):
        box = [torch.zeros_like(i[..., 0:4]) for i in inference]
        obj = [torch.zeros_like(i[..., 4]) for i in inference]
        cls = [torch.zeros_like(i[..., 5:][bi, ai, cj, ci]) for i, (bi, ci, cj, ai) in zip(inference, indices)]
        return box, obj, cls

    @staticmethod
    def extract_inference(inference, indices):
        box = [i[..., 0:4] for i in inference]
        obj = [i[..., 4] for i in inference]
        cls = [i[..., 5:][bi, ai, cj, ci] for i, (bi, ci, cj, ai) in zip(inference, indices)]
        return box, obj, cls

    @staticmethod
    def area(a, indices=(0, 1, 2, 3)):
        x0, y0, x1, y1 = indices
        return (a[..., x1] - a[..., x0]) * (a[..., y1] - a[..., y0])

    @staticmethod
    def xywh2rect(xywh, indices=(0, 1, 2, 3)):
        x, y, w, h = indices
        rect = xywh.clone()
        rect[..., x] = xywh[..., x] - xywh[..., w] / 2
        rect[..., y] = xywh[..., y] - xywh[..., h] / 2
        rect[..., w] = xywh[..., x] + xywh[..., w] / 2
        rect[..., h] = xywh[..., y] + xywh[..., h] / 2
        return rect

    @staticmethod
    def make_anchors(outputs):
        s, _, _ = outputs.anchors.shape
        return torch.cat(
            (
                torch.empty_like(outputs.anchors).fill_(0.5),
                outputs.anchors / outputs.strides.view(s, 1, 1)
            ), dim=2
        )  # (s x a x 4)

    @property
    def obj_pos_weight(self):
        return self._obj_pos_weight

    @obj_pos_weight.setter
    def obj_pos_weight(self, value):
        self._obj_pos_weight = torch.tensor(value)
        self.obj_bce = BCEWithLogitsLoss(pos_weight=self._obj_pos_weight)

    @property
    def cls_pos_weight(self):
        return self._cls_pos_weight

    @cls_pos_weight.setter
    def cls_pos_weight(self, value):
        self._cls_pos_weight = torch.tensor(value)
        self.cls_bce = BCEWithLogitsLoss(pos_weight=self._cls_pos_weight)
