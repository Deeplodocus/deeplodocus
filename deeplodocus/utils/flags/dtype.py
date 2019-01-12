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
DEEP_DTYPE_IMAGE = Flag(name="Image", description="Image type", names=["image", "images", "img"])
DEEP_DTYPE_VIDEO = Flag(name="Video", description="Video type", names=["video", "videos", "vid"])
DEEP_DTYPE_INTEGER = Flag(name="Int", description="Integer type", names=["integer", "int"])
DEEP_DTYPE_FLOAT = Flag(name="Float", description="Float type", names=["float", "flt"])
DEEP_DTYPE_BOOLEAN = Flag(name="Bool", description="Boolean type", names=["boolean", "bool"])
DEEP_DTYPE_AUDIO = Flag(name="Sound", description="Sound type", names=["audio", "sound", "mp3", "cda"])
DEEP_DTYPE_SEQUENCE = Flag(name="Sequence", description="Sequence type", names=["sequence"])
DEEP_DTYPE_NP_ARRAY = Flag(name="Numpy", description="Numpy array type",
                           names=["numpy", "npy", "npz", "numpy array", "numpy_array", "numpy-array"])

