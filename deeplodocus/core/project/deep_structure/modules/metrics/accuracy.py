import numpy as np


def accuracy(out, labels):
    out = out.detach().numpy()
    labels = labels.detach().numpy()
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)/float(labels.size)
