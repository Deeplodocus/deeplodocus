"""
Authors : Alix Leroy,
Test the loading of a dataset
"""
import time

from deeplodocus.data.dataset import DataSet

inputs = []
labels = []
additional_data = []
inputs.append([r"input1.txt", r"input2.txt"])
inputs.append([r"./images", r"./images"])
labels.append([r"label1.txt", r"label2.txt"])
labels.append(r"label3.txt")

additional_data.append(r"label3.txt")
#additional_data.append(r"additional_data.txt")

dataset = DataSet(inputs, labels, additional_data, transform=None,  cv_library="opencv", write_logs=False)
dataset.fill_data()
dataset.summary()
#inputs, labels, additional_data = dataset.__getitem__(1)


#inputs, labels = dataset.__getitem__(1)
num_images = 32

t0 = time.time()
for i in range(num_images//2):
    inputs, labels, additional_data = dataset.__getitem__(1)

t1 = time.time()
print(t1-t0)

#print(inputs[0])
#print(inputs[1])


import cv2
t2 = time.time()
for i in range(num_images):
    image =  cv2.imread("./images/image2.png", cv2.IMREAD_ANYDEPTH)
t3 = time.time()
print(t3-t2)


print("Time (s) to load " + str(num_images) + " images using DEEPLODOCUS : " +str(t1-t0) + ". Average per image :"+ str((t1-t0) / num_images)     )
print("Time (s) to load " + str(num_images) + " images using RAW OPENCV function : " +str(t3-t2) + ". Average per image :"+ str((t3-t2) / num_images)     )


