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

DEEP_SHUFFLE_RANDOM_PICK = Flag(
    name="Pick random indices",
    description="Pick randomly indices in the list available",
    names=["pick", "random_pick", "random pick", "random-pick"]
)

