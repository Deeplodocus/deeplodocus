import numpy as np

from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import *


class ConfusionMatrix(object):

    def __init__(self, filename="ConfusionMatrix", output_index=0):
        self.output_index = output_index
        self.filename = filename
        self.n_classes = None
        self.matrix = None

    def forward(self, outputs=None, labels=None):
        label = labels[:, self.output_index].cpu().numpy()
        output = outputs[self.output_index].cpu().numpy()
        if self.n_classes is None:
            self.n_classes = output.shape[1]
            self.matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
        output = np.argmax(output, axis=1)
        label = self.one_hot_encode(label, self.n_classes).T.reshape(self.n_classes, 1, -1)
        output = self.one_hot_encode(output, self.n_classes).T.reshape(1, self.n_classes, -1)
        self.matrix += np.sum(label * output, axis=2).astype(int)
        return outputs

    @staticmethod
    def one_hot_encode(batch, depth):
        return np.eye(depth)[batch].astype(np.uint8)

    def finish(self):
        filepath = "%s.csv" % self.filename
        np.savetxt(filepath, self.matrix, fmt="%i", delimiter=",")
        Notification(DEEP_NOTIF_SUCCESS, "confusion matrix saved to : %s" % filepath)
