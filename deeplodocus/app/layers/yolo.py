import torch
import torch.nn as nn


class YoloLayer(nn.Module):

    def __init__(self, num_classes, image_shape, num_anchors=3, device=None):
        super(YoloLayer, self).__init__()
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.image_shape = image_shape

    def forward(self, x):
        batch_size, _, input_height, input_width = x.size
        stride = (
            self.image_shape[0] / input_width,
            self.image_shape[1] / input_height
        )
        prediction = x.view(
            batch_size,
            self.num_anchors,
            self.num_classes + 5,
            input_height,
            input_width
        ).permute(0, 1, 3, 4, 2).contiguous()

        # Outputs
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        obj = prediction[..., 4]
        cls = prediction[..., 5:]

        # Calculate offsets
        grid_x = torch.arange(input_width).repeat(input_width).view(1, 1, input_height, input_width)
        grid_y = torch.arange(input_height).repeat(input_height).view(1, 1, input_height, input_width)
        scaled_anchors = [(a_w / stride[1], a_h / stride[2]) for a_w, a_h in self.anchors]
        anchor_w = scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

        # Add offset and scale with anchors
        if self.device.type == "cuda":
            boxes = torch.cuda.FloatTensor(prediction[..., :4].shape)
        else:
            boxes = torch.FloatTensor(prediction[..., :4].shape)

        boxes[..., 0] = x.data + grid_x
        boxes[..., 1] = y.data + grid_y
        boxes[..., 2] = torch.exp(w.data) * anchor_w
        boxes[..., 3] = torch.exp(h.data) * anchor_h

        output = torch.cat(
            (
                boxes.view(batch_size, -1, 4) * stride,
                obj.view(batch_size, -1, 1),
                cls.view(batch_size, -1, self.num_classes),
            ),
            -1,
        )
        return output
