import cv2

from deeplodocus.data.transform.transform_data import TransformData
from deeplodocus.app.transforms.yolo.label import reformat, scale


def reformat_pointer(x, **kwargs):
    kwargs["image_shape"] = x.shape[0:2]
    return x, TransformData(
        name="reformat",
        method=reformat,
        module_path="modules.transforms.label",
        kwargs=kwargs
    )


def resize(image, shape):
    image = cv2.resize(image, tuple(shape))
    return image, TransformData(
        name="scale",
        module_path="modules.transforms.label",
        method=scale,
        kwargs={"image_shape": shape}
    )
