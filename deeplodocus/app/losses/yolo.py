import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import numpy as np
import cv2
import time

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

    def compress_targets(self, targets):
        b, n, _ = targets.shape
        batch_index = torch.arange(b, device=targets.device).repeat_interleave(n)  # (b * n)
        targets = torch.cat((batch_index.view(b, n, 1).float(), targets), dim=2).view(-1, 6)
        return targets[~torch.all(targets[:, 1:6] == -1, dim=1)]

    def make_target_boxes(self, targets, strides):
        target_boxes = targets.view(-1, 1, 4).repeat(1, len(strides), 1) / strides.view(1, 3, 1)
        c = torch.floor(target_boxes[..., 0:2])
        target_boxes[..., 0:2] -= c
        return target_boxes, c.long()

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


if __name__ == "__main__":

    class Outputs:
        def __init__(self):
            pass

    def reformat_targets(targets):
        new_targets = torch.zeros(4, 20, 5)
        new_targets.fill_(-1)
        i = 0
        last_batch = 0
        for t in targets:
            b = int(t[0])
            if b > last_batch:
                i = 0
            new_targets[b, i] = t[1:]
            i += 1
            last_batch = b
        return new_targets


    def inference(p, anchor_wh, stride):
        bs, _, ny, nx, _ = p.shape
        grid = create_grids((nx, ny)).cuda()
        io = p.clone()  # inference output
        io[..., :2] = torch.sigmoid(io[..., :2]) + grid  # xy
        io[..., 2:4] = torch.exp(io[..., 2:4]) * anchor_wh  # wh yolo method
        io[..., :4] *= stride
        torch.sigmoid_(io[..., 4:])
        return io.view(bs, -1, 85), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]

    def create_grids(ng=(13, 13), device='cpu'):
        nx, ny = ng
        yv, xv = torch.meshgrid([torch.arange(ny, device=device), torch.arange(nx, device=device)])
        grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        return grid


    # Get images
    imgs = torch.load("/home/samuel/YOLOTest/imgs.pt").cpu().permute(0, 2, 3, 1)[..., [2, 1, 0]].contiguous()
    imgs = (imgs.numpy() * 255).astype(np.uint8)

    # Get predictions
    p = [torch.load("/home/samuel/YOLOTest/p_%i.pt" % i) for i in range(3)][::-1]

    # Get anchors and strides
    anchors = torch.tensor(
        (
            ((10, 13), (16, 30), (33, 23)),
            ((30, 61), (62, 45), (59, 119)),
            ((116, 90), (156, 198), (373, 326))
        )
    ).float().cuda()
    strides = torch.tensor((8, 16, 32)).cuda()

    # Init loss cacluator
    loss = YOLOLoss(iou_threshold=0.2, giou=True)

    # Construct model outputs
    outputs = Outputs()
    outputs.__dict__["anchors"] = anchors
    outputs.__dict__["inference"] = p
    outputs.__dict__["strides"] = strides

    # Format targets
    targets = torch.load("/home/samuel/YOLOTest/targets.pt")
    targets[:, 2:6] *= 512
    targets = reformat_targets(targets).cuda()

    # Scaled anchors
    scaled_anchors = [torch.load("/home/samuel/YOLOTest/anchors_%i.pt" % i).view(1, 3, 1, 1, 2) for i in range(3)]

    # Show images with labels
    if False:
        images = np.zeros((1024, 1024, 3), dtype=np.uint8)
        for n, (img, labels) in enumerate(zip(imgs, targets.cpu().numpy())):
            for lab in labels[~np.all(labels == -1, axis=1)]:
                p1 = tuple((lab[1:3] - lab[3:5] / 2).astype(int))
                p2 = tuple((lab[1:3] + lab[3:5] / 2).astype(int))
                img = cv2.rectangle(np.array(img), p1, p2, (20, 200, 20), 1)
            i = n % 2
            j = int(n / 2)
            images[j * 512: (j + 1) * 512, i * 512: (i + 1) * 512, ] = img
        cv2.imshow("images", images)
        cv2.waitKey(0)

    # calculate loss
    loss = loss.forward(outputs, targets)

    quit()
    # Display inference
    images = np.zeros((1024, 1024, 3), dtype=np.uint8)
    p = [inference(i, a, s)[0] for i, a, s in zip(p[::-1], scaled_anchors, (32, 16, 8))]
    p = torch.cat(p, dim=1)
    for n, (img, pred) in enumerate(zip(imgs, p)):
        pred = pred[pred[:, 4] > 0.5]
        pred[:, 5] = torch.argmax(pred[:, 5:], dim=1)
        pred = pred[:, 0:6]
        for lab in pred.detach().cpu().numpy():
            p1 = tuple((lab[0:2] - lab[2:4] / 2).astype(int))
            p2 = tuple((lab[0:2] + lab[2:4] / 2).astype(int))
            img = cv2.rectangle(np.array(img), p1, p2, (20, 200, 20), 1)
        i = n % 2
        j = int(n / 2)
        images[j * 512: (j + 1) * 512, i * 512: (i + 1) * 512, ] = img
        cv2.imshow("inference", images)
    cv2.waitKey(0)

