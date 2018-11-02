import numpy as np

def accuracy(out, labels):
  outputs = np.argmax(out, axis=1)
  return np.sum(outputs==labels)/float(labels.size)