from deeplodocus.utils.flag import Flag

# LIST POSSIBLE SOURCES

DEEP_SOURCE_FILE = Flag("Source : local file", ["file"])
DEEP_SOURCE_FOLDER = Flag("Source : local folder", ["folder"])
DEEP_SOURCE_DATABASE = Flag("Source : database", ["database", "db"])
DEEP_SOURCE_SERVER = Flag("Source : server", ["server", "remote"])
DEEP_SOURCE_SPARK = Flag("Source: Spark access", ["spark"])
