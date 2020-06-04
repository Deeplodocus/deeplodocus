from deeplodocus.utils.flag import Flag

#
# SHUFFLE
#
DEEP_SHUFFLE_NONE = Flag(
    name="No shuffling",
    description="No shuffling",
    names=["none", "no", "false"]
)
DEEP_SHUFFLE_BATCHES = Flag(
    name="Batch shuffling",
    description="Batch shuffling",
    names=["batches", "batch", "shuffle batches", "shuffle_batches", "shuffle-batches"]
)
DEEP_SHUFFLE_ALL = Flag(
    name="Shuffle all",
    description="Shuffling all the dataset",
    names=["all", "default", "shuffle all", "shuffle_all", "shuffle-all"]
)
