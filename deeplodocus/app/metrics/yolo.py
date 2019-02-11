import torch
import torch.nn as nn


class AveragePrecision(nn.Module):

    def __init__(self, iou_threshold=0.5, conf_threshold=0.25):
        super(AveragePrecision, self).__init__()
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def forward(self, outputs, targets):
        metric = torch.tensor(0.0)
        for output, anchors in outputs:                                     # For outputs and anchors from each layer
            batch_size, num_anchors, h, w, v = output.shape
            for b in range(batch_size):
                # Targets from current batch and remove targets with all zeros
                target = targets[b, :, :]
                target = target[~torch.all(target == 0, dim=1)]

                # Flattened predictions
                prediction = output[b, :, :, :, ].view(-1, v)

                # Prediction = x, y, w, h, obj, class, obj * class score
                prediction = torch.cat(
                    (
                        prediction[:, 0:5],
                        torch.argmax(prediction[5:]),
                        torch.max(prediction[:, 5:])[0] * prediction[:, 4],
                    ), 1
                )

                # Remove predictions with a low confidence
                prediction = prediction[prediction[:, 6] > self.conf_threshold]


