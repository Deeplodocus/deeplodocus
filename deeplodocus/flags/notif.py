from deeplodocus.utils.flag import Flag

#
# NOTIFICATION FLAGS
#

DEEP_NOTIF_INFO = Flag(
    name="Info notification",
    description="Blue info notification",
    names=["info"]
)
DEEP_NOTIF_DEBUG = Flag(
    name="Debug notification",
    description="Purple debug notification",
    names=["debug"]
)
DEEP_NOTIF_SUCCESS = Flag(
    name="Success notification",
    description="Green success notification",
    names=["success"]
)
DEEP_NOTIF_WARNING = Flag(
    name="Warning notification",
    description="Orange warning notification",
    names=["warning"]
)
DEEP_NOTIF_ERROR = Flag(
    name="Error notification",
    description="Red error notification",
    names=["error"]
)
DEEP_NOTIF_FATAL = Flag(
    name="Fatal notification",
    description="White fatal notification on red background",
    names=["fatal"]
)
DEEP_NOTIF_INPUT = Flag(
    name="Input notification",
    description="Blanking white result notification",
    names=["input"]
)
DEEP_NOTIF_RESULT = Flag(
    name="Result notification",
    description="White result notification",
    names=["result"]
)
DEEP_NOTIF_LOVE = Flag(
    name="Love notification",
    description="Pink love notification",
    names=["love", "luv", "<3"]
)
