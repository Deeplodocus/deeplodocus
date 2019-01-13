from deeplodocus.utils.flag import Flag

#
# EVENT TYPES
#

DEEP_EVENT_UNDEFINED = Flag(name= "Undefined", description="Event : Undefined", names=["none", "undefined"])
DEEP_EVENT_ON_BATCH_END = Flag(name= "On Batch End", description="Event : On Batch End", names=["batch end", "end batch", "end_batch", "on_batch_end"])
DEEP_EVENT_ON_EPOCH_END = Flag(name= "On Epoch End", description="Event : On Epoch End", names=["epoch end", "end epoch", "on epoch end"])
DEEP_EVENT_ON_TRAINING_START = Flag(name= "On Training Start", description="Event : On Training Start", names=["training start", "on training start"])
DEEP_EVENT_ON_TRAINING_END = Flag(name= "On Training End", description="Event : On Training End", names=["training end", "on training end"])
DEEP_EVENT_ON_UPDATE_ALL = Flag(name= "On Update all", description="Event : On Update All", names=["update all"])
DEEP_EVENT_ON_UPDATE_MODEL = Flag(name= "On Update Model", description="Event : On Update Model", names=["update model"])
DEEP_EVENT_ON_UPDATE_OPTIMIZER = Flag(name= "On Update Optimizer", description="Event : On Update Optimizer", names=["update optimizer"])
DEEp_EVENT_ON_UPDATE_DATASET = Flag(name= "On Update Dataset", description="Event : On Update Dataset", names=["update dataset"])
DEEP_EVENT_END_LISTENING = Flag(name= "End Listening", description="Event : End Listening", names=["end listening"])
DEEP_EVENT_OVERWATCH_METRIC_COMPUTED = Flag(name= "Overwatch metric computed", description="Event : Overwatch Metric Computed", names=["overwatch", "overwatch metric", "overwatch metric computed"])
DEEP_EVENT_ON_EPOCH_START = Flag(name= "On Epoch Start", description="Event : On Epoch Start", names=["on epoch start"])
DEEP_EVENT_SAVING_REQUIRED = Flag(name= "Is Saving Required", description="Event : Is Saving Required", names=["is saving required"])
DEEP_EVENT_SAVE_MODEL = Flag(name= "Save Model", description="Transformer : Pointer", names=["save model", "save_model"])