from deeplodocus.utils.flag import Flag
#
# VERBOSE
#
DEEP_VERBOSE_BATCH = Flag(
    name="Verbose Batch",
    description="Print information at the end of every batch",
    names=["batch", "batches", "default"]
)
DEEP_VERBOSE_EPOCH = Flag(
    name="Verbose Epoch",
    description="Print information at the end of every epoch",
    names=["epoch", "epochs"]
)
DEEP_VERBOSE_TRAINING = Flag(
    name="Verbose Training",
    description="Print information at the end of the training",
    names=["training"]
)
