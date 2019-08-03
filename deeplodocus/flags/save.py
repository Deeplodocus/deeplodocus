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
