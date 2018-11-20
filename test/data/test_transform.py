from deeplodocus.data.transform_manager import TransformManager
from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.flags import *

# Get the config for the transform managers
config_transforms = Namespace("./transforms")


# Create the transform managers
transform_manager_train = TransformManager(config_transforms.train)
#transform_manager_val = TransformManager(config_transforms.validation, write_logs=False)
#transform_manager_test = TransformManager(config_transforms.test, write_logs=False)

import time
from deeplodocus.data.dataset import Dataset
from deeplodocus.utils.types import *
from PIL import Image
import numpy as np
import cv2


inputs = []
labels = []
additional_data = []
inputs.append([r"input1.txt", r"input2.txt"])
inputs.append([r"./images", r"./images"])
labels.append([r"label1.txt", r"label2.txt"])
#labels.append(r"label3.txt")
#labels = []

additional_data.append(r"label3.txt")
#additional_data.append(r"additional_data.txt")
additional_data = []


dataset = Dataset(inputs, labels, additional_data, transform_manager=transform_manager_train,  cv_library=DEEP_LIB_OPENCV, name="Test")
dataset.load()
dataset.set_len_dataset(1000)
dataset.summary()
#inputs, labels, additional_data = dataset.__getitem__(1)

t0 = time.time()
inputs, labels, additional_data = dataset.__getitem__(500)
t1 = time.time()

print(t1-t0)
print(labels)
print(len(inputs))
cv2.imshow("test", inputs[1])
cv2.waitKey(0)


