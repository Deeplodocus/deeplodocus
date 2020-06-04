
def rect2xywh(rect, indices=(0, 1, 2, 3)):
    a, b, c, d = indices
    rect = rect.T
    xywh = rect.copy()
    xywh[a] = (rect[a] + rect[c]) / 2
    xywh[b] = (rect[b] + rect[d]) / 2
    xywh[c] = rect[c] - rect[a]
    xywh[d] = rect[d] - rect[b]
    return xywh.T, None


def normalize_boxes(box, image_shape, indices=(0, 1, 2, 3)):
    a, b, c, d = indices
    box = box.T
    box[a] /= image_shape[0]
    box[b] /= image_shape[1]
    box[c] /= image_shape[0]
    box[d] /= image_shape[1]
    return box.T, None
