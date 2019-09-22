from deeplodocus.utils.flag import Flag


#
# LOAD_AS FLAGS
#
DEEP_LOAD_AS_STRING = Flag(
    name="String",
    description="String type",
    names=["string", "str"]
)
DEEP_LOAD_AS_IMAGE = Flag(
    name="Image",
    description="Image type",
    names=["image", "images", "img"]
)

DEEP_LOAD_AS_VIDEO = Flag(
    name="Video",
    description="Video type",
    names=["video", "videos", "vid"]
)

DEEP_LOAD_AS_INTEGER = Flag(
    name="Int",
    description="Integer type",
    names=["integer", "int"]
)
DEEP_LOAD_AS_FLOAT = Flag(
    name="Float",
    description="Float type",
    names=["float", "flt"]
)
DEEP_LOAD_AS_BOOLEAN = Flag(
    name="Bool",
    description="Boolean type",
    names=["boolean", "bool"]
)
DEEP_LOAD_AS_AUDIO = Flag(
    name="Sound",
    description="Sound type",
    names=["audio", "sound", "mp3", "cda"]
)

DEEP_LOAD_AS_SEQUENCE = Flag(
    name="Sequence",
    description="Sequence type",
    names=["sequence"])

DEEP_LOAD_AS_NP_ARRAY = Flag(
    name="Numpy",
    description="Numpy array type",
    names=["numpy", "npy", "npz", "numpy array", "numpy_array", "numpy-array"]
)
