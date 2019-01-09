from deeplodocus.utils.flag import Flag

#
# SHUFFLE
#
DEEP_SHUFFLE_NONE = Flag("No shuffling", names=["none"])
DEEP_SHUFFLE_BATCHES = Flag("Batch shuffling", names=["batches", "batch"])
DEEP_SHUFFLE_ALL = Flag("Shuffling all the dataset", names=["all"])