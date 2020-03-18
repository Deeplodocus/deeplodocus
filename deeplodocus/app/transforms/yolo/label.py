import numpy as np

from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import DEEP_NOTIF_WARNING


def reformat(x, image_shape, n_obj=100):
    # Initialize label array
    label = np.empty((n_obj, 5), dtype=np.float32)
    label.fill(-1)

    # If the input array is empty, return the empty label array
    if not np.any(x):
        return label, None

    # Enumerate over the list of object labels
    for i, item in enumerate(x):
        # If the array does not have enough rows, print a warning and break
        if i >= n_obj:
            Notification(
                DEEP_NOTIF_WARNING,
                "label.reformat : Unable to fit all %i objects into label array : consider increasing n_obj" % len(x)
            )
            break
        # Put the box and category id into the array
        # NB: Coco format is x0, y0, w, h (where x0,y0 = upper left corner)
        # NB: we want x, y, w, h (where x,y = centre)
        try:
            box = np.array(item["bbox"])
        except IndexError:
            Notification(DEEP_NOTIF_WARNING, "Unusual label instance : %s : shape%s" % (type(x), x.shape))
            return label, None
        label[i, 0:2] = box[0:2] + box[2:4] / 2
        label[i, 2:4] = box[2:4]
        label[i, 4] = item["category_id"]

    # Normalise box labels w.r.t. image height and width
    label[:i + 1, 0:4] /= np.tile(np.array(image_shape)[[1, 0]], 2).reshape(1, -1)
    return label, None
