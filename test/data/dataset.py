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

dataset = DataSet(inputs, labels, additional_data, transform=None,  cv_library="PIL", write_logs=False)
dataset.fill_data()
dataset.summary()
inputs, labels, additional_data = dataset.__getitem__(1)


#inputs, labels = dataset.__getitem__(1)

t0 = time.time()
for i in range(2):
    for j in range(14):
        #inputs, labels = dataset.__getitem__(1)
        inputs, labels, additional_data = dataset.__getitem__(1)

t1 = time.time()
print(t1-t0)