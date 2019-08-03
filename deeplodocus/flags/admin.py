from deeplodocus.utils.flag import Flag

DEEP_ADMIN_START_PROJECT = Flag(
    name="Start Project",
    description="startproject : Start a deeplodocus project",
    names=["start_project", "startproject"]
)

DEEP_ADMIN_VERSION = Flag(
    name="Version",
    description="version : Display Deeplodocus Version",
    names=["version"]
)

DEEP_ADMIN_HELP = Flag(
    name="Help",
    description="help : Display the Deeplodocus commands",
    names=["help"]
)