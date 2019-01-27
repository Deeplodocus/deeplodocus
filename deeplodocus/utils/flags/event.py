from deeplodocus.utils.flag import Flag

#
# EVENT TYPES
#

DEEP_EVENT_UNDEFINED = Flag(name="Undefined",
                            description="Event : Undefined",
                            names=["none", "undefined"])

DEEP_EVENT_ON_BATCH_END = Flag(name="On Batch End",
                               description="Event : On Batch End",
                               names=["batch end", "end batch", "end_batch", "on_batch_end"])

DEEP_EVENT_ON_EPOCH_END = Flag(name="On Epoch End",
                               description="Event : On Epoch End",
                               names=["epoch end", "end epoch", "on epoch end"])

DEEP_EVENT_ON_TRAINING_START = Flag(name="On Training Start",
                                    description="Event : On Training Start",
                                    names=["training start", "on training start"])

DEEP_EVENT_ON_TRAINING_END = Flag(name="On Training End",
                                  description="Event : On Training End",
                                  names=["training end", "on training end"])

DEEP_EVENT_ON_UPDATE_ALL = Flag(name="On Update all",
                                description="Event : On Update All",
                                names=["update all"])

DEEP_EVENT_ON_UPDATE_MODEL = Flag(name="On Update Model",
                                  description="Event : On Update Model",
                                  names=["update model"])

DEEP_EVENT_ON_UPDATE_OPTIMIZER = Flag(name="On Update Optimizer",
                                      description="Event : On Update Optimizer",
                                      names=["update optimizer"])

DEEp_EVENT_ON_UPDATE_DATASET = Flag(name="On Update Dataset",
                                    description="Event : On Update Dataset",
                                    names=["update dataset"])

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
DEEP_EVENT_ON_EPOCH_START = Flag(
    name="On Epoch Start",
    description="Event : On Epoch Start",
    names=["on epoch start"]
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


#
# MODEL SIGNALS
#
DEEP_SAVE_SIGNAL_END_BATCH = Flag(name="End of batch",
                                  description="Save the model after each mini-batch",
                                  names=["batch",
                                         "onbatch",
                                         "on batch",
                                         "on-batch",
                                         "on_batch",
                                         "endbatch",
                                         "end batch",
                                         "end-batch",
                                         "end_batch"])

DEEP_SAVE_SIGNAL_END_EPOCH = Flag(name="End of epoch",
                                  description="Save the model after each epoch",
                                  names=["epoch",
                                         "epochend",
                                         "epoch end",
                                         "epoch-end",
                                         "epoch_end",
                                         "endepoch",
                                         "end epoch",
                                         "end-epodh",
                                         "end_epoch"])

DEEP_SAVE_SIGNAL_END_TRAINING = Flag(name="End of training",
                                     description="Save at the end of training",
                                     names=["training",
                                            "endtraining",
                                            "end training"
                                            "end-training",
                                            "end_training",
                                            "trainingend",
                                            "training end",
                                            "training-end",
                                            "training_end"])

DEEP_SAVE_SIGNAL_AUTO = Flag(name="Auto",
                             description="Save the model when the evaluation metric is better than all previous values",
                             names=["auto", "default"])

