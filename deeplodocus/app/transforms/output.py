import numpy as np

from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import *


class ConfusionMatrix(object):

    def __init__(self, filename="ConfusionMatrix.csv", fill=5):
        self.filename = filename
        self.n_classes = None
        self.matrix = None
        self.fill = fill

    def forward(self, outputs=None, labels=None):
        label = labels.cpu().numpy()
        output = outputs.cpu().numpy()
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

    def print(self):
        Notification(DEEP_NOTIF_RESULT, "┌" + "─" * ((self.fill + 3) * self.matrix.shape[1] - 3) + "┐")
        for row in self.matrix:
            Notification(DEEP_NOTIF_RESULT, "│" + " : ".join([str(i).rjust(self.fill, " ") for i in row]) + "│")
        Notification(DEEP_NOTIF_RESULT, "└" + "─" * ((self.fill + 3) * self.matrix.shape[1] - 3) + "┘")

    def finish(self):
        Notification(DEEP_NOTIF_RESULT, "Confusion matrix :")
        self.print()
        np.savetxt(self.filename, self.matrix, fmt="%i", delimiter=",")
        Notification(DEEP_NOTIF_SUCCESS, "Confusion matrix saved to : %s" % self.filename)
