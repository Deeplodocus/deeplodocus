#!/usr/bin/env python3

import os
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import Callback


class History(Callback):

    def __init__(self, metrics, initial_epoch="auto", log_dir="../logs", file_name="history.csv"):
        Callback.__init__(self)
        self.log_dir = log_dir
        self.file_path = "%s/%s" % (log_dir, file_name)
        self.metrics = metrics
        self.history = pd.DataFrame(columns=["wall time", "relative time", "epoch"] + metrics)
        self.start_time = 0
        self.epoch = initial_epoch
        self.__load_history()
        self.__set_initial_epoch()

    def on_train_begin(self, logs=None):
        """
        :param logs:
        :return:
        """
        self.__set_start_time()

    def on_epoch_end(self, epoch, logs=None):
        """
        :param epoch:
        :param logs:
        :return:
        """
        self.epoch += 1
        data = dict(list(zip(self.metrics, [logs.get(metric) for metric in self.metrics])))
        data["wall time"] = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S")
        data["relative time"] = self.__time()
        data["epoch"] = self.epoch
        self.history = self.history.append(data, ignore_index=True)

    def on_train_end(self, logs=None):
        self.history.to_csv(self.file_path, index=False)

    def plot(self):
        x = self.history["epoch"]
        labels = ["Loss", "Accuracy"]
        colors = ["blue", "red"]
        plots = [["loss", "val_loss"], ["acc", "val_acc"]]
        n = len(plots)
        for i, (label, sub_plot) in enumerate(zip(labels, plots)):
            plt.subplot(n, 1, i + 1)
            for metric, col, in zip(sub_plot, colors):
                plt.plot(x, self.history[metric], c=col, label=metric)
            plt.legend()
            plt.xlabel("Epoch")
            plt.ylabel(label)
            plt.grid()
        plt.show()

    def __load_history(self):
        if os.path.isfile(self.file_path):
            self.history = pd.read_csv(self.file_path)

    def __set_start_time(self):
        if self.history.empty:
            self.start_time = time.time()
        else:
            self.start_time = time.time() - self.history["relative time"][self.history.index[-1]]

    def __set_initial_epoch(self):
        if self.epoch == "auto":
            if self.history.empty:
                self.epoch = 0
            else:
                self.epoch = self.history["epoch"][self.history.index[-1]]

    def __time(self):
        """
        :return: Current train time in seconds
        """
        return round(time.time() - self.start_time, 2)
