from deeplodocus.utils.flag import Flag

#
# DATA MEMORIZATION CONDITION
#

DEEP_MEMORIZE_BATCHES = Flag("Memorize batches", description="Memorize info at each batch", names=["default", "batch", "batches"])
DEEP_MEMORIZE_EPOCHS = Flag("Memorize epochs", description="Memorize info at each epoch", names=["epoch", "epochs"])