#!/usr/bin/env python3

#
# ONLY kept the interesting part from previous file
#
import matplotlib.pyplot as plt


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
