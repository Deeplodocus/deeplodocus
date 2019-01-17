from deeplodocus.utils.flag import Flag

#
# SAVE FORMATS
#
DEEP_SAVE_FORMAT_ONNX = Flag(name="ONNX",
                             description="Open Neural Network eXchange format",
                             names=["onnx"])
DEEP_SAVE_FORMAT_PYTORCH = Flag(name="PyTorch",
                                description="Saving with Python's pickle module",
                                names=["pytorch", "pt", "pth", "default"])


#
# SAVE CONDITIONS
#

DEEP_SAVE_CONDITION_LESS = Flag(name="Less than",
                                description="Call saver when given metric is smaller than all previous values",
                                names=["<", "smaller", "less", "default"])

DEEP_SAVE_CONDITION_GREATER = Flag(name="Greater than",
                                   description="Call saver when given metric is greater than all previous values",
                                   names=[">", "bigger", "greater"])
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
