import cv2
import contextlib
import numpy as np
import torch
from torchvision.ops import nms

from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import DEEP_NOTIF_WARNING


class Activate(object):

    def __init__(self, skip=1, initial_skip=0):
        """
        :param skip: int: number of batches to skip between visualizing
        :param initial_skip: int: the number of batches to skip initially
        """
        self.skip = skip
        self.initial_skip = initial_skip
        self._batch = 0
        self._total_batches = 0

    def __repr__(self):
        return "<Activate Transform Object>"

    def forward(self, outputs):
        """
        Concatenates all yolo from different scales
        :param outputs: dict: yolo output
        :return: dict: yolo output, with detections reformatted to (b x ? x 5 + num_cls)
        """
        self._batch += 1
        self._total_batches += 1
        if (not self._batch % self.skip) and self._total_batches >= self.initial_skip:
            outputs.inference = torch.cat(
                [
                    self.__forward(x, anchors=anchors, stride=stride)
                    for x, anchors, stride in
                    zip(outputs.inference, outputs.anchors, outputs.strides)
                ], dim=1
            )
        return outputs

    @staticmethod
    def __forward(x, anchors, stride):
        b, a, h, w, n = x.shape
        # Make grids of cell locations
        cx = torch.arange(w, device=x.device).repeat(h, 1).view([1, 1, h, w]).float()
        cy = torch.arange(h, device=x.device).repeat(w, 1).t().view([1, 1, h, w]).float()
        # Apply transforms to bx, by, bw, bh
        x[..., 0] = (torch.sigmoid(x[..., 0]) + cx) * stride
        x[..., 1] = (torch.sigmoid(x[..., 1]) + cy) * stride
        x[..., 2:4] = anchors.view(1, a, 1, 1, 2) * torch.exp(x[..., 2:4])
        x[..., 4:] = torch.sigmoid(x[..., 4:])
        return x.view(b, -1, n)

    def finish(self):
        self._batch = 0


class NMS(object):

    def __init__(self, skip=1, initial_skip=0, iou_threshold=0.5, obj_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.obj_threshold = obj_threshold
        self.skip = skip
        self.initial_skip = initial_skip
        self._batch = 0
        self._total_batches = 0

    def forward(self, outputs):
        self._batch += 1
        self._total_batches += 1
        if (not self._batch % self.skip) and self._total_batches >= self.initial_skip:
            for b, batch in enumerate(outputs.inference):
                t, n = batch.shape
                batch = batch[batch[:, 4] > self.obj_threshold]
                indices = nms(xywh2rect(batch[:, 0:4]), batch[:, 4], self.iou_threshold)
                k = torch.zeros([batch.shape[0]] * 2, device=batch.device)
                k[:indices.shape[0]] = torch.eye(batch.shape[0], device=batch.device)[indices]
                batch = torch.mm(k, batch)
                batch[torch.all(k == 0)] = -1
                new_batch = torch.zeros(t, n, device=batch.device, dtype=batch.dtype).fill_(-1)
                new_batch[0:batch.shape[0]] = batch
                outputs.inference[b] = new_batch
        return outputs


class Visualize(object):

    def __init__(
            self,
            window_name="YOLO Visualization",
            obj_threshold=0.5,
            scale=1.0,
            skip=1,
            wait=1,
            rows=None,
            cols=None,
            width=1,
            key=None,
            lab_col=(32, 200, 32),
            det_col=(32, 32, 200),
            font_scale=1.0,
            font_thickness=1,
            initial_skip=0
    ):
        """
        :param window_name: str: name for the cv2 window
        :param scale: float: to scale the output visualisation up or down
        :param skip: int: number of batches to skip between visualizing
        :param wait: int: number of ms to wait after displaying window (0 = wait for key press)
        :param rows: int: number of rows when stitching the batch of images into a single large image
        :param cols: int: number of cols when stitching the batch of images into a single large image
        :param width: int: line width for drawing boxes
        :param lab_col: tuple(ints): the color for drawing label
        :param det_col: tuple(ints): the color for the output label
        :param initial_skip: int: the number of batches to skip initially
        """
        self.window_name = window_name
        self.obj_threshold = obj_threshold
        self.scale = scale
        self.rows = rows
        self.cols = cols
        self.skip = skip
        self.wait = wait
        self.width = width
        self.lab_col = tuple(lab_col)
        self.det_col = tuple(det_col)
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.initial_skip = initial_skip
        self._total_batches = 0
        self._batch = 0
        self.key = key
        if key is not None:
            with open(key, "r") as file:
                self.key = [i.strip() for i in file.readlines() if i.strip()]

    def __repr__(self):
        return "<Visualize Transform Object>"

    def forward(self, inputs, outputs, labels=None):
        """
        Author: SW
        :param inputs: torch tensor: network inputs
        :param outputs: torch tensor: network outputs
        :param labels: torch tensor: instance labels (optional)
        :return:
        """
        self._batch += 1
        self._total_batches += 1
        if (not self._batch % self.skip) and self._total_batches >= self.initial_skip:
            if inputs is None:
                inputs = np.zeros((16, 512, 512, 3), dtype=np.uint8)
            else:
                inputs = inputs[0].permute((0, 2, 3, 1)).cpu().numpy().astype(np.uint8)  # Convert to numpy
                inputs = np.ascontiguousarray(inputs[:, :, :, [2, 1, 0]])  # Convert to bgr
            detections = outputs.inference.detach().cpu().clone().numpy()  # Get detections as numpy
            # Draw labels
            if labels is not None:
                labels = labels.cpu().numpy()
                inputs = self.__draw_boxes(inputs, labels, color=self.lab_col, width=self.width)
            # Remove predictions where objectness score < obj_threshold
            detections[detections[..., 4] < self.obj_threshold] = -1
            # Draw model predictions
            inputs = self.__draw_boxes(inputs, detections, color=self.det_col, width=self.width)
            images = self.__stitch_images(inputs)  # Stitch all images into one large image
            # Rescale
            if self.scale != 1:
                images = cv2.resize(images, (0, 0), fx=self.scale, fy=self.scale)
            # Display
            cv2.imshow(self.window_name, images)
            cv2.waitKey(self.wait)
        return outputs

    def finish(self):
        """
        Author: SW
        Called at the end of an epoch : reset batches and destroy all windows
        :return: None
        """
        self._batch = 0
        with contextlib.suppress(cv2.error):
            cv2.destroyWindow(self.window_name)

    def __stitch_images(self, inputs):
        """
        Author: SW
        Stitches the given batch of images into a single image
        :param inputs: np array: batch of images (b x h x w x ch)
        :return: np array: single large image (H, W, ch)
        """
        rows, cols = self.__get_rows_cols(inputs.shape[0])
        h, w, ch = inputs.shape[-3:]
        images = np.zeros((h * rows, w * cols, ch), dtype=np.uint8)
        for k, image in enumerate(inputs):
            j = int(k / cols)
            i = k % cols
            images[h * j:h * (j + 1), w * i:w * (i + 1), :] = image
        return images

    def __draw_boxes(self, images, boxes, color=(32, 200, 32), width=1):
        """
        Author: SW
        Draw the given boxes on the given images
        :param images: np array: array of images to draw boxes on (b x h x w x c)
        :param boxes: np array: array of boxes to draw (b x n x 4) (xywh format)
        :param color: tuple(ints): color for the boxes
        :param width: int: width of the box lines
        :return: images with given boxes drawn
        """
        for image, batch in zip(images, boxes):
            batch = batch[~ np.all(batch == -1, axis=1)]
            classes = np.argmax(batch[:, 5:], axis=1) if batch.shape[1] > 5 else batch[:, 4]
            for rect, cls in zip(xywh2rect(batch[:, 0:4]), classes.astype(int)):
                cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[2:4]), color, width)
                cv2.putText(
                    image,
                    str(cls) if self.key is None else self.key[cls],
                    tuple(rect[0:2]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    color,
                    self.font_thickness
                )
                #except TypeError:
                #    Notification(
                #        DEEP_NOTIF_WARNING,
                #        "Unable to draw rectangle : p1=%s : p2=%s" % (tuple(rect[0:2]), tuple(rect[2:4]))
                #    )
        return images

    def __get_rows_cols(self, n):
        """
        Author: SW
        This function returns the number of rows and columns required for the display of each image
        If rows and cols are not given, they are inferred from n
        If just rows is given, col = n/rows
        If just cols is given, rows = n/cols
        If both rows and cols are given, use those values
        :param n: int: batch size
        :return: int, int: rows, cols
        """
        # If rows and cols are both given
        if self.rows is not None and self.cols is not None:
            rows = int(self.rows)
            cols = int(self.cols)
        # If rows is given
        elif self.rows is not None:
            rows = int(self.rows)
            cols = int(np.ceil(n / rows))
        # If cols is given
        elif self.cols is not None:
            cols = int(self.rows)
            rows = int(np.ceil(n / cols))
        # If neither rows or cols are given
        else:
            # If the n is a square number, rows = sqrt(n)
            if np.sqrt(n) % 1 == 0:
                rows = int(np.sqrt(n))
            # Otherwise, make twice as many cols as rows
            else:
                rows = int(np.sqrt(n / 2))
            # Infer cols from n and rows
            cols = int(np.ceil(n / rows))
        if rows * cols < n:
            Notification(
                DEEP_NOTIF_WARNING,
                "yolo/output/Visualize : Not all images are displayed : n=%i, rows=%i, cols=%i" % (n, rows, cols)
            )
        return rows, cols


def xywh2rect(array, indices=(0, 1, 2, 3)):
    """
    Author: SW
    Convert a given array of box coordinates from xywh to upper and lower corner coordinates
    :param array: torch or np array: given array to convert
    :param indices: tuple(ints): the indices of the last dimension for x, y, w, h respectively
    :return: converted coordinates
    """
    x, y, w, h = indices
    if isinstance(array, np.ndarray):
        output = array.copy()
    else:
        output = array.clone()
    output[..., x] = array[..., x] - array[..., w] / 2
    output[..., y] = array[..., y] - array[..., h] / 2
    output[..., w] = array[..., x] + array[..., w] / 2
    output[..., h] = array[..., y] + array[..., h] / 2
    return output
