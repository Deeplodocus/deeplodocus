import numpy as np


def nms(prediction, conf_threshold=0.25, iou_threshold=0.5):
    batch_size, a, h, w, v = prediction.shape

    # Flatten prediction
    prediction = prediction[batch_size, :, :, :, ].view(-1, v)

    # Prediction = x, y, w, h, obj, class, obj * class score
    prediction = np.concatenate(
        (
            prediction[:, 0:5],
            np.argmax(prediction[5:]),
            np.max(prediction[:, 5:]) * prediction[:, 4],
        ), 1
    )

    # Remove predictions with a low confidence
    prediction = prediction[prediction[:, 6] > conf_threshold]

    output = ()
    for b in range(batch_size):
        to_suppress = []
        current = prediction[b, np.argmax(prediction[:, 6])],
        compare = prediction[b, prediction[:, 5] == current[5]]
        ious = iou(xywh2box(current), xywh2box(compare))[:, iou != 1]


def xywh2box(xywh):
    """
    :param xywh:
    :return:
    """
    box = np.empty_like(xywh.T)
    box[0] = xywh.T[0] - xywh.T[2] / 2
    box[1] = xywh.T[1] - xywh.T[3] / 2
    box[2] = xywh.T[0] + xywh.T[2] / 2
    box[3] = xywh.T[1] + xywh.T[3] / 2
    return box.T


def area(box):
    return (box.T[2] - box.T[0]) * (box.T[3] - box.T[1])


def iou(box, boxes):
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    flag = np.all(np.stack((x0 < x1, y0 < y1), axis=1), axis=1)
    union = (x1 - x0) * (y1 - y0) * flag
    return union / (area(box) + area(boxes) - union)
