from deeplodocus.utils.flag import Flag

#
# POSSIBLE SOURCES
#

DEEP_SOURCE_FILE = Flag(name="File",
                        description="Source : local file",
                        names=["file"])

DEEP_SOURCE_FOLDER = Flag(name="Folder",
                          description="Source : local folder",
                          names=["folder", "dir", "directory"])

DEEP_SOURCE_DATABASE = Flag(name="Database",
                            description="Source : database",
                            names=["database", "db"])

DEEP_SOURCE_SERVER = Flag(name="Server",
                          description="Source : server",
                          names=["server", "remote"])

DEEP_SOURCE_SPARK = Flag(name="Spark",
                         description="Source: Spark access",
                         names=["spark"])

DEEP_SOURCE_PREMADE_DATASET = Flag(name="Pre-made dataset",
                                   description="Source: A pre-made dataset",
                                   names=["dataset", "premade", "pre-made"])
