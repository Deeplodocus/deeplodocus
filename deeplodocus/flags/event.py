from deeplodocus.utils.flag import Flag


# INFERENCE EVENTS
DEEP_EVENT_UNDEFINED = Flag(
    name="Undefined",
    description="Event : Undefined",
    names=["none", "undefined"]
)
DEEP_EVENT_BATCH_START = Flag(
    name="Batch Start",
    description="Event : Batch Start",
    names=["batch start", "batch-start", "batch_start", "end start", "end-start", "end_start"]
)
DEEP_EVENT_BATCH_END = Flag(
    name="Batch End",
    description="Event : Batch End",
    names=["batch end", "batch-end", "batch_end", "end batch", "end-batch", "end_batch"]
)
DEEP_EVENT_EPOCH_START = Flag(
    name="Epoch Start",
    description="Event : Epoch Start",
    names=["epoch start", "epoch-start", "epoch_start", "start epoch", "start-epoch", "start_epoch"]
)
DEEP_EVENT_EPOCH_END = Flag(
    name="Epoch End",
    description="Event : Epoch End",
    names=["epoch end", "epoch-end", "epoch_end", "end epoch", "end-epoch", "end_epoch"]
)
DEEP_EVENT_VALIDATION_START = Flag(
    name="Validation Start",
    description="Event : Validation Start",
    names=[
        "val start", "val-start", "val_start", "start val", "start-val", "start_val",
        "validation start", "validation-start", "validation_start",
        "start validation", "start-validation", "start_validation"
    ]
)
DEEP_EVENT_VALIDATION_END = Flag(
    name="Validation End",
    description="Event : Validation End",
    names=[
        "val end", "val-end", "val_end", "end val", "end-val", "end_val",
        "validation end", "validation-end", "validation_end",
        "end validation", "end-validation", "end_validation"
    ]
)
DEEP_EVENT_TRAINING_START = Flag(
    name="Training Start",
    description="Event : Training Start",
    names=[
        "training start", "training-start", "training_start",
        "start training", "start-training", "start_training",
        "train start", "train-start", "train_start",
        "start train", "start-train", "start_train"
    ]
)
DEEP_EVENT_TRAINING_END = Flag(
    name="Training End",
    description="Event : Training End",
    names=[
        "training end", "training-end", "training_end",
        "end training", "end-training", "end_training",
        "train end", "train-end", "train_end",
        "end train", "end-train", "end_train"
    ]
)

# SAVER EVENTS
DEEP_SAVE_SIGNAL_END_BATCH = Flag(
    name="End of batch",
    description="Save the model after each mini-batch",
    names=[
        "batch", "onbatch", "on batch", "on-batch",  "on_batch",
        "endbatch", "end batch", "end-batch", "end_batch"
    ]
)
DEEP_SAVE_SIGNAL_END_EPOCH = Flag(
    name="End of epoch",
    description="Save the model after each epoch",
    names=["epoch", "epochend", "epoch end", "epoch-end", "epoch_end", "endepoch", "end epoch", "end_epoch"]
)
DEEP_SAVE_SIGNAL_END_TRAINING = Flag(
    name="End of training",
    description="Save at the end of training",
    names=[
        "training", "endtraining", "end training", "end-training", "end_training",
        "trainingend", "training end", "training-end", "training_end"
    ]
)
DEEP_SAVE_SIGNAL_AUTO = Flag(
    name="Auto",
    description="Save the model when the evaluation metric is better than all previous values",
    names=["auto", "default", "overwatch", "over_watch", "over-watch", "over watch"]
)

# OTHER EVENTS
DEEP_EVENT_ON_UPDATE_ALL = Flag(
    name="On Update all",
    description="Event : On Update All",
    names=["update all"]
)
DEEP_EVENT_ON_UPDATE_MODEL = Flag(
    name="On Update Model",
    description="Event : On Update Model",
    names=["update model"]
)
DEEP_EVENT_ON_UPDATE_OPTIMIZER = Flag(
    name="On Update Optimizer",
    description="Event : On Update Optimizer",
    names=["update optimizer"]
)
DEEp_EVENT_ON_UPDATE_DATASET = Flag(
    name="On Update Dataset",
    description="Event : On Update Dataset",
    names=["update dataset"]
)
DEEP_EVENT_END_LISTENING = Flag(
    name="End Listening",
    description="Event : End Listening",
    names=["end listening"]
)
DEEP_EVENT_OVERWATCH_METRIC_COMPUTED = Flag(
    name="Overwatch metric computed",
    description="Event : Overwatch Metric Computed",
    names=["overwatch", "overwatch metric", "overwatch metric computed"]
)
DEEP_EVENT_SAVING_REQUIRED = Flag(
    name="Is Saving Required",
    description="Event : Is Saving Required",
    names=["is saving required"]
)
DEEP_EVENT_SAVE_MODEL = Flag(
    name="Save Model",
    description="Transformer : Pointer",
    names=["save model", "save_model"]
)
DEEP_EVENT_REQUEST_TRAINING_LOSS = Flag(
    name="Request training loss",
    description="Request training loss",
    names=["request_training_loss"]
)
DEEP_EVENT_SEND_TRAINING_LOSS = Flag(
    name="Send training loss",
    description="Send training loss",
    names=["send_training_loss"]
)
DEEP_EVENT_REQUEST_SAVE_PARAMS_FROM_TRAINER = Flag(
    name="Request save params",
    description="Request save params",
    names=["request_save_params"]
)
DEEP_EVENT_SEND_SAVE_PARAMS_FROM_TRAINER = Flag(
    name="Send save params",
    description="Send save params",
    names=["send_save_params"]
)

