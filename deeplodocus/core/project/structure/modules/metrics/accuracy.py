import torch


# def accuracy(inputs, labels):
#     _, inputs = inputs.max(1)
#     return torch.mean((inputs == labels).float())

import numpy as np


def Accuracy(out, labels):
    out = out.detach().numpy()
    labels = labels.detach().numpy()
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)/float(labels.size)
