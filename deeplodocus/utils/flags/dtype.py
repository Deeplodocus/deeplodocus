from deeplodocus.utils.flag import Flag

#
# TYPE FLAGS
#
DEEP_TYPE_FILE = 0
DEEP_TYPE_FOLDER = 1
DEEP_TYPE_IMAGE = 2
DEEP_TYPE_VIDEO = 3
DEEP_TYPE_INTEGER = 4
DEEP_TYPE_FLOAT = 5
DEEP_TYPE_BOOLEAN = 6
DEEP_TYPE_SOUND = 7
DEEP_TYPE_SEQUENCE = 8
DEEP_TYPE_NP_ARRAY = 9

#
# DATA TYPE FLAGS
#
DEEP_DTYPE_IMAGE = Flag(description="Image type", names=["image", "images", "img"])
DEEP_DTYPE_VIDEO = Flag(description="Video type", names=["video", "videos", "vid"])
DEEP_DTYPE_INTEGER = Flag(description="Integer type", names=["integer", "int"])
DEEP_DTYPE_FLOAT = Flag(description="Float type", names=["float", "flt"])
DEEP_DTYPE_BOOLEAN = Flag(description="Boolean type", names=["boolean", "bool"])
DEEP_DTYPE_AUDIO = Flag(description="Sound type", names=["audio", "sound", "mp3", "cda"])
DEEP_DTYPE_SEQUENCE = Flag(description="Sequence type", names=["sequence"])
DEEP_DTYPE_NP_ARRAY = Flag(description="Numpy array type", names=["numpy", "npy", "npz", "numpy array", "numpy_array", "numpy-array"])

