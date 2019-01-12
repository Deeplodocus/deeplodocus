from deeplodocus.utils.flag import Flag
#
# LIBRARIES
#
#DEEP_LIB_PIL = 0
#DEEP_LIB_OPENCV = 1

DEEP_LIB_PIL = Flag(name="PIL", description="Computer vision library : PIL", names=["pil"])
DEEP_LIB_OPENCV = Flag(name="OpenCV", description="Computer vision library : OpenCV", names=["opencv"])
