import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_history(history_dir, line_width=0.5, alpha=1, y_scale="linear"):
    history_files = os.listdir(history_dir)
    batch_data= None
    epoch_data = None
    val_data = None
    for file in history_files:
        if "train_batches" in file:
            batch_data = pd.read_csv("/".join((history_dir, file)))
        elif "train_epochs" in file:
            epoch_data = pd.read_csv("/".join((history_dir, file)))
        elif "validation" in file:
            val_data = pd.read_csv("/".join((history_dir, file)))
    # Plot training data for every batch
    if batch_data is not None:
        col_names = list(batch_data)
        num_batches = max(batch_data["Batch"])
        t = batch_data["Epoch"] - 1 + (batch_data["Batch"] - 1) / num_batches
        # Plot sub losses
        plt.plot(
            t,
            batch_data["Total Loss"],
            label="Total Loss (training)",
            linewidth=line_width
        )
        # Plot total loss
        for i in range(5, batch_data.shape[1]):
            plt.plot(
                t,
                batch_data.iloc[:, i],
                label=col_names[i] + " (training)",
                linewidth=line_width,
                alpha=alpha
            )
    # Plot training data for each epoch (only if batch data is None)
    elif epoch_data is not None:
        print("Plotting for training epoch not implemented yet")
    # Plot validation data for each epoch
    if val_data is not None:
        col_names = list(val_data)
        # Plot total validation loss
        plt.plot(
            val_data["Total Loss"],
            label="Total Loss (validation)",
            linewidth=line_width
        )
        # Plot validation sub-losses
        for i in range(4, val_data.shape[1]):
            plt.plot(
                val_data.iloc[:, i],
                label=col_names[i] + " (validation)",
                linewidth=line_width,
                alpha=alpha
            )
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.yscale(y_scale)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_history("/home/samuel/Cranfield University/Code/deeplodocus-dev/test/core/YOLOv3/v03/history",
                 alpha=0.5,
                 y_scale="linear")
