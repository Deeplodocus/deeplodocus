from deeplodocus.utils.flag import Flag

DEEP_ADMIN_START_PROJECT = Flag(
    name="Start Project",
    description="startproject : Start a deeplodocus project",
    names=["start_project", "startproject", "start-project"]
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

DEEP_ADMIN_TRANSFORMER = Flag(
    name="Transformer",
    description="Create a template transformer file",
    names=["transformer"]
)

DEEP_ADMIN_OUTPUT_TRANSFORMER = Flag(
    name="Output Transformer",
    description="Create a template output transformer file",
    names=["outputtransformer", "output-transformer", "output_transformer"]
)

DEEP_ADMIN_ONEOF_TRANSFORMER = Flag(
    name="One-of Transformer",
    description="Create a template one-of input transformer file",
    names=["oneoftransformer", "oneof-transformer", "oneof_transformer", "one-of-transformer", "one_of_transformer"]
)

DEEP_ADMIN_SEQUENTIAL_TRANSFORMER = Flag(
    name="Sequential Transformer",
    description="Create a template sequential input transformer file",
    names=["sequentialtransformer", "sequential-transformer", "sequential_transformer"]
)

DEEP_ADMIN_SOMEOF_TRANSFORMER = Flag(
    name="Sequential Transformer",
    description="Create a template sequential input transformer file",
    names=["someoftransformer", "someof-transformer", "someof_transformer", "some-of-transformer", "some_of_transformer"]
)

