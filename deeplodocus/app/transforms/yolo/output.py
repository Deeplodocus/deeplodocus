import cv2
import contextlib
import numpy as np
import torch

from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import DEEP_NOTIF_WARNING


class Concatenate(object):

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
        return "<Concatenate Transform Object>"

    def forward(self, outputs):
        """
        Concatenates all yolo from different scales
        :param outputs: dict: yolo output
        :return: dict: yolo output, with detections reformatted to (b x ? x 5 + num_cls)
        """
        self._batch += 1
        self._total_batches += 1
        if (not self._batch % self.skip) and self._total_batches >= self.initial_skip:
            for i in range(len(outputs["detections"])):
                b, _, _, _, n = outputs["detections"][i].shape
                outputs["detections"][i][..., 0:4] *= outputs["strides"][i]
                outputs["detections"][i] = outputs["detections"][i].view(b, -1, n)
            outputs["detections"] = torch.cat(outputs["detections"], 1)
        return outputs

    def finish(self):
        self._batch = 0


class NonMaximumSuppression(object):

    def __init__(self, num_types=4, num_classes=10, iou_threshold=0.5, obj_threshold=0.5, skip=1, initial_skip=0):
        self.iou_threshold = iou_threshold
        self.obj_threshold = obj_threshold
        self.num_types = num_types
        self.num_classes = num_classes
        self.skip = skip
        self.initial_skip = initial_skip
        self._batch = 0
        self._total_batches = 0

    def __repr__(self):
        return "<NonMaximumSupression Transform Object>"

    @staticmethod
    def __area(tensor):
        dims = list(range(tensor.dim()))
        dims.reverse()
        return (tensor.permute(*dims)[0] - tensor.permute(*dims)[2]) * (tensor.permute(*dims)[1] - tensor.permute(*dims)[3])

    def __iou(self, active, others):
        x0 = torch.max(others[:, 0], active[0])
        y0 = torch.max(others[:, 1], active[1])
        x1 = torch.min(others[:, 2], active[2])
        y1 = torch.min(others[:, 3], active[3])
        mask = (x0 < x1) * (y0 < y1)
        intersection = (x1 - x0) * (y1 - y0) * mask.type(active.dtype)
        return intersection / (self.__area(active) + self.__area(others) - intersection)

    @staticmethod
    def __xywh2rect(xywh):
        dims = list(range(xywh.dim()))
        dims.reverse()
        rect = torch.zeros(xywh.shape, dtype=xywh.dtype).permute(*dims)
        rect = rect.to(device=xywh.device)
        rect[0] = xywh.permute(*dims)[0] - xywh.permute(*dims)[2] / 2
        rect[1] = xywh.permute(*dims)[1] - xywh.permute(*dims)[3] / 2
        rect[2] = xywh.permute(*dims)[0] + xywh.permute(*dims)[2] / 2
        rect[3] = xywh.permute(*dims)[1] + xywh.permute(*dims)[3] / 2
        return rect.permute(*dims)

    @staticmethod
    def __compress_classes(outputs):
        compressed_outputs = outputs[:, 0:6]
        with contextlib.suppress(RuntimeError):
            _, cls = outputs[:, 5:].max(1)
            compressed_outputs[:, 5] = cls
        return compressed_outputs

    def __obj_supression(self, outputs):
        return outputs[outputs[:, 4] > self.obj_threshold]

    def __non_maximum_supression(self, outputs):
        n = outputs.shape[-1]
        suppressed_outputs = []
        while outputs.shape[0]:
            _, i = outputs[:, 4].max(0)                     # Get index of output with highest obj score
            current = outputs[i, :]                         # Set as current
            suppressed_outputs.append(current)              # Append current to list
            ious = self.__iou(
                self.__xywh2rect(current[0:4]),
                self.__xywh2rect(outputs[:, 0:4])
            )                                               # Get iou scores between current and and all outputs
            outputs = outputs[ious < self.iou_threshold]    # Remove outputs which overlap by more than the threshold
        if suppressed_outputs:
            return torch.stack(suppressed_outputs, 0)
        else:
            return torch.tensor((), device=outputs.device).view(-1, n)

    def __forward(self, outputs):
        # Remove outputs where object score is less than obj_threshold
        outputs = self.__obj_supression(outputs)
        # Change class and type predictions from one-hot encoded to a single value
        outputs = self.__compress_classes(outputs)
        # Perform non-maximum suppression
        outputs = self.__non_maximum_supression(outputs)
        return outputs

    def forward(self, outputs):
        self._batch += 1
        self._total_batches += 1
        if (not self._batch % self.skip) and self._total_batches >= self.initial_skip:
            # Apply sigmoid to objectness scores
            outputs["detections"][..., 4] = torch.sigmoid(outputs["detections"][..., 4])
            b, n, _ = outputs["detections"].shape
            # Initialise a tensor to put detections in
            detections = torch.zeros((b, n, 6), device=outputs["detections"].device)
            # Perform nms on each batch
            for i, output in enumerate(outputs["detections"]):
                dets = self.__forward(output)
                detections[i, 0:dets.shape[0], :] = dets
            # Minimise size of second dimension
            # Because the number of detections should be much smaller than n
            outputs["detections"] = detections[:, ~ torch.all(detections.view(-1, n) == 0, dim=0)]
        return outputs

    def finish(self):
        self._batch = 0


class Visualize(object):

    def __init__(
            self,
            window_name="YOLO Visualization",
            scale=1.0,
            skip=1,
            wait=1,
            rows=None,
            cols=None,
            width=1,
            lab_col=(32, 200, 32),
            out_col=(32, 32, 200),
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
        :param out_col: tuple(ints): the color for the output label
        :param initial_skip: int: the number of batches to skip initially
        """
        self.window_name = window_name
        self.scale = scale
        self.rows = rows
        self.cols = cols
        self.skip = skip
        self.wait = wait
        self.width = width
        self.lab_col = tuple(lab_col)
        self.out_col = tuple(out_col)
        self.initial_skip = initial_skip
        self._total_batches = 0
        self._batch = 0

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
            inputs = inputs[0].permute((0, 2, 3, 1)).cpu().numpy().astype(np.uint8)     # Convert to numpy
            inputs = np.ascontiguousarray(inputs[:, :, :, [2, 1, 0]])                   # Convert to bgr
            predictions = outputs["detections"].detach().cpu().clone().numpy()

            # Draw labels
            if labels is not None:
                labels = labels.cpu().clone().numpy()
                inputs = self.__draw_boxes(inputs, labels, color=self.lab_col, width=self.width, scale=True)

            # Draw model predictions
            inputs = self.__draw_boxes(inputs, predictions, color=self.out_col, width=self.width)

            # Stitch all images into one large image
            images = self.__stitch_images(inputs)

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

    @staticmethod
    def __draw_boxes(images, boxes, color=(32, 200, 32), width=1, scale=False):
        """
        Author: SW
        Draw the given boxes on the given images
        :param images: np array: array of images to draw boxes on (b x h x w x c)
        :param boxes: np array: array of boxes to draw (b x n x 4) (xywh format)
        :param color: tuple(ints): color for the boxes
        :param width: int: width of the box lines
        :param scale: bool: if the boxes need to be scaled up or not
        :return: images with given boxes drawn
        """
        if scale:
            # Make bbox values absolute
            boxes = scale_boxes(boxes, images.shape)
        for image, rects in zip(images, xywh2rect(boxes).astype(int)):
            for box in rects:
                if not np.all(box < 0):
                    cv2.rectangle(image, tuple(box[0:2]), tuple(box[2:4]), color, width)
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


def scale_boxes(boxes, shape):
    """
    Author: SW
    Scale the given bounding boxes by the given shape
    :param boxes: np array: given bounding boxes
    :param shape: tuple(ints/floats): shape to scale by (h, w)
    :return: np array: scaled bounding boxes
    """
    boxes[..., 0:4] *= np.tile(np.array(shape[1:3])[[1, 0]], 2).reshape(1, 1, -1)
    return boxes


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
