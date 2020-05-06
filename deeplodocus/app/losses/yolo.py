import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss


class YOLOLoss(nn.Module):

    def __init__(self, iou_threshold=0.5, obj_weight=[0.5, 2], box_weight=5, class_weight=None):
        super(YOLOLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.obj_weight = torch.tensor(obj_weight)
        self.box_weight = box_weight
        self.bce = BCEWithLogitsLoss(reduction=False)
        self.class_weight = class_weight
        self._cls_freq = None

    def forward(self, outputs, targets):
        s, a, _ = outputs.anchors.shape
        b, n, _ = targets.shape
        num_classes = outputs.inference[0].shape[4] - 5
        targets = targets.view(-1, 5)
        mask = ~torch.all(targets == -1, dim=1)  # (b * n) mask of targets to ignore
        batch_index = torch.arange(b, device=targets.device).repeat_interleave(n)  # (b * n)
        anchors = torch.cat(
            (
                torch.empty_like(outputs.anchors).fill_(0.5),
                outputs.anchors / outputs.strides.view(s, 1, 1)
            ), dim=2
        )  # (s x a x 4)
        target_box = targets[:, 0:4].clone().view(-1, 1, 4).repeat(1, s, 1)
        target_box[mask] /= outputs.strides.view(1, s, 1)  # (b * n x s x 4)
        c = target_box[..., 0:2].long()  # (b * n x s x 2)
        target_box[mask, :, 0:2] -= c[mask]
        iou = jaccard_index(
            xywh2rect(target_box).view(-1, s, 1, 4),
            xywh2rect(anchors).view(1, s, a, 4)
        )  # (b * n x s x a)
        best_prior = torch.eye(
            s * a, dtype=torch.bool, device=targets.device  # (b * n x s x a)
        )[torch.argmax(iou.view(-1, s * a), dim=1)].view(-1, s, a)  # (b * n x s x a)
        overlap_prior = torch.mul(iou > self.iou_threshold, ~ best_prior)  # (b * n x s x a)
        scale_mask = torch.any(best_prior, dim=2)  # (n * b x s)
        anchor_index = torch.argmax(torch.any(best_prior, dim=1).float(), dim=1)  # (b * n)
        target_box[mask, :, 0:2] = logit(target_box[mask, :, 0:2])
        target_box = target_box.view(-1, s, 1, 4).repeat(1, 1, a, 1)
        target_box[mask, :, :, 2:4] = log(target_box[mask, :, :, 2:4] / anchors[..., 2:4].view(1, s, a, 2))
        target_obj = best_prior.float() - overlap_prior.float()
        target_cls = torch.eye(num_classes, device=targets.device)[targets[:, 4].long()]
        gt_box, gt_obj, gt_cls = self.__initialise_gt(outputs.inference)
        for i in range(s):
            # Box gt
            m = torch.mul(mask, scale_mask[:, i])
            gt_box[i].index_put_(
                indices=tuple(
                    torch.cat((batch_index[m, None], anchor_index[m, None], c[m, i][:, [1, 0]]), dim=1).T
                ),
                values=target_box[best_prior][m]
            )
            # Obj gt
            index_obj = torch.cat((batch_index[mask, None], c[mask, i][:, [1, 0]]), dim=1)
            gt_obj[i] = gt_obj[i].permute(0, 2, 3, 1)
            gt_obj[i].index_put_(indices=tuple(index_obj.T), values=target_obj[mask, i])
            gt_obj[i] = gt_obj[i].permute(0, 3, 1, 2)
            # Cls gt
            m = torch.add(overlap_prior[:, i], best_prior[:, i]).long()
            gt_cls[i] = gt_cls[i].permute(0, 2, 3, 1, 4)
            gt_cls[i].index_put_(
                indices=tuple(index_obj.T),
                values=(
                        target_cls[:, None, :] * m[:, :, None]
                        - torch.ones_like(target_cls[:, None, :]) * (1 - m[:, :, None])
                )[mask]
            )
            gt_cls[i] = gt_cls[i].permute(0, 3, 1, 2, 4)

        class_weight = self.__update_class_weights(targets[mask, 4], num_classes)
        inf_box, inf_obj, inf_cls = self.__extract_inference(outputs.inference)
        box_loss = self.__caclculate_box_loss(gt_box, inf_box)
        obj_loss = self.__calculate_obj_loss(gt_obj, inf_obj)
        cls_loss = self.__calculate_cls_loss(gt_cls, inf_cls, class_weight)
        return box_loss + obj_loss + cls_loss

    def __caclculate_box_loss(self, gt, inf):
        gt = torch.cat([i.view(-1, 4) for i in gt])
        inf = inf[~torch.all(gt == -1, dim=1)]
        gt = gt[~torch.all(gt == -1, dim=1)]
        return self.box_weight * torch.mean(torch.sum((gt - inf) ** 2, dim=1))

    def __calculate_obj_loss(self, gt, inf):
        gt = torch.cat([i.view(-1) for i in gt])
        inf = inf[gt != -1]
        gt = gt[gt != -1]
        weight = self.obj_weight[gt.long()].to(gt.device)
        return torch.mean(self.bce(inf, gt) * weight)

    def __calculate_cls_loss(self, gt, inf, class_weight):
        gt = torch.cat([i.view(-1, i.shape[-1]) for i in gt])
        inf = inf[~torch.any(gt == -1, dim=1)]
        gt = gt[~torch.any(gt == -1, dim=1)]
        weight = class_weight[torch.argmax(gt, dim=1)].to(gt.device)
        return torch.mean(self.bce(inf, gt) * weight.view(-1, 1))

    @staticmethod
    def __initialise_gt(inference):
        box = [torch.empty_like(i[..., 0:4]).fill_(-1) for i in inference]
        obj = [torch.zeros_like(i[..., 4]) for i in inference]
        cls = [torch.zeros_like(i[..., 5:]).fill_(-1) for i in inference]
        return box, obj, cls

    @staticmethod
    def __extract_inference(inference):
        box = torch.cat([i[..., 0:4].view(-1, 4) for i in inference])
        obj = torch.cat([i[..., 4].view(-1) for i in inference])
        cls = torch.cat([i[..., 5:].view(-1, i.shape[-1] - 5) for i in inference])
        return box, obj, cls

    def __update_class_weights(self, targets, num_classes):
        if self.class_weight is None:
            return torch.ones(num_classes)
        elif self.class_weight.lower() == "auto":
            if self._cls_freq is None:
                self._cls_freq = torch.zeros(num_classes)
            self._cls_freq.index_put_(
                indices=tuple(targets.view(1, -1).long().cpu()),
                values=torch.tensor(1.0),
                accumulate=True
            )
            weight = torch.sum(self._cls_freq) / (num_classes * self._cls_freq)
            weight[weight < 1] = 1
            return weight
        else:
            return torch.tensor(self.class_weight)


def xywh2rect(xywh, indices=(0, 1, 2, 3)):
    """
    :param xywh:
    :param indices:
    :return:
    """
    x, y, w, h = indices
    rect = xywh.clone()
    rect[..., x] = xywh[..., x] - xywh[..., w] / 2
    rect[..., y] = xywh[..., y] - xywh[..., h] / 2
    rect[..., w] = xywh[..., x] + xywh[..., w] / 2
    rect[..., h] = xywh[..., y] + xywh[..., h] / 2
    return rect


def jaccard_index(a1, a2):
    """
    :param a1:
    :param a2:
    :return:
    """
    x0 = torch.max(a1[..., 0], a2[..., 0])
    y0 = torch.max(a1[..., 1], a2[..., 1])
    x1 = torch.min(a1[..., 2], a2[..., 2])
    y1 = torch.min(a1[..., 3], a2[..., 3])
    mask = (x0 < x1) * (y0 < y1)
    intersection = (x1 - x0) * (y1 - y0) * mask.type(torch.float32)
    return intersection / (area(a1) + area(a2) - intersection)


def area(a, indices=(0, 1, 2, 3)):
    """
    :param a:
    :param indices:
    :return:
    """
    x0, y0, x1, y1 = indices
    return (a[..., x1] - a[..., x0]) * (a[..., y1] - a[..., y0])


def logit(x, e=1e-3):
    """
    logit function
    :param x: torch.tensor
    :return: logit(x)
    """
    x[x == 0] = e
    x[x == 1] = 1 - e
    return torch.log(x / (1 - x))


def log(x, e=1e-3):
    """
    """
    x[x < e] = e
    return torch.log(x)
