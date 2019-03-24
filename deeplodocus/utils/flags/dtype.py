from deeplodocus.utils.flag import Flag


#
# DATA TYPE FLAGS
#
DEEP_DTYPE_STRING = Flag(
    name="String",
    description="String type",
    names=["string", "str"]
)
DEEP_DTYPE_IMAGE = Flag(
    name="Image",
    description="Image type",
    names=["image", "images", "img"]
)
DEEP_DTYPE_IMAGE_INT = Flag(
    name="ImageInt",
    description="Image type with integers",
    names=["imageint", "image-int", "img-int", "img_int", "image_int"]
)
DEEP_DTYPE_VIDEO = Flag(
    name="Video",
    description="Video type",
    names=["video", "videos", "vid"]
)

DEEP_DTYPE_INTEGER = Flag(
    name="Int",
    description="Integer type",
    names=["integer", "int"]
)
DEEP_DTYPE_FLOAT = Flag(
    name="Float",
    description="Float type",
    names=["float", "flt"]
)
DEEP_DTYPE_BOOLEAN = Flag(
    name="Bool",
    description="Boolean type",
    names=["boolean", "bool"]
)
DEEP_DTYPE_AUDIO = Flag(
    name="Sound",
    description="Sound type",
    names=["audio", "sound", "mp3", "cda"]
)
DEEP_DTYPE_SEQUENCE = Flag(
    name="Sequence",
    description="Sequence type",
    names=["sequence"])
DEEP_DTYPE_NP_ARRAY = Flag(
    name="Numpy",
    description="Numpy array type",
    names=["numpy", "npy", "npz", "numpy array", "numpy_array", "numpy-array"]
)
