"""
Authors : Alix Leroy,
Test the loading of a dataset
"""
import time
from deeplodocus.data.dataset import Dataset
from deeplodocus.utils.types import *
from PIL import Image
import numpy as np


inputs = []
labels = []
additional_data = []
inputs.append([r"input1.txt", r"input2.txt"])
inputs.append([r"./images", r"./images"])
labels.append([r"label1.txt", r"label2.txt"])
labels.append(r"label3.txt")
labels = []

additional_data.append(r"label3.txt")
#additional_data.append(r"additional_data.txt")
additional_data = []


dataset = Dataset(inputs, labels, additional_data, transform_manager=None,  cv_library=DEEP_OPENCV, write_logs=False, name="Test")
dataset.load()
dataset.summary()
#inputs, labels, additional_data = dataset.__getitem__(1)


#inputs, labels = dataset.__getitem__(1)
num_images = 100


t0 = time.time()
for i in range(num_images//2):
    #inputs, labels, additional_data = dataset.__getitem__(1
    inputs = dataset.__getitem__(1)
t1 = time.time()
print(t1-t0)



import cv2
t2 = time.time()
for i in range(num_images):
    image =  cv2.imread("./images/image2.png", cv2.IMREAD_UNCHANGED)# If the image is not a grayscale (only width + height axis)
t3 = time.time()
print(t3-t2)




t4 = time.time()
for i in range(num_images):
    image = Image.open("./images/image2.png")
    image = np.array(image)
t5 = time.time()
print(t5-t4)




print("Time (s) to load " + str(num_images) + " images using DEEPLODOCUS : " +str(t1-t0) + ". Average per image :"+ str((t1-t0) / num_images)     )
print("Time (s) to load " + str(num_images) + " images using RAW OPENCV function : " +str(t3-t2) + ". Average per image :"+ str((t3-t2) / num_images)     )
print("Time (s) to load " + str(num_images) + " images using RAW PIL function : " +str(t5-t4) + ". Average per image :"+ str((t5-t4) / num_images)     )


