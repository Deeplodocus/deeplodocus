import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_history(history_dir, line_width=0.5, alpha=1.0, scale="linear", grid=True, batches=True, validation=True):
    history_files = os.listdir(history_dir)
    if batches and "history_train_batches.csv" in history_files:
        plot_training_batches(history_dir, line_width=line_width, alpha=alpha)
    if (not batches or "history_train_batches.csv" not in history_files) \
            and "history_train_epochs.csv" in history_files:
        plot_training_epochs(history_dir, line_width=line_width, alpha=alpha)
    if validation and "history_validation.csv" in history_files:
        plot_validation_history(history_dir, line_width=line_width)
    plt.yscale(scale)
    if grid:
        plt.grid()
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.show()


def plot_training_epochs(history_dir, line_width=0.5, alpha=1.0):
    df = pd.read_csv("/".join((history_dir, "history_train_epochs.csv")))
    col_names = list(df)
    plt.plot(
        df["Epoch"], df["Total Loss"],
        label="Total Loss (training)",
        linewidth=line_width
    )
    # Plot validation sub-losses
    for i in range(4, df.shape[1]):
        plt.plot(
            df["Epoch"], df.iloc[:, i],
            label=col_names[i] + " (training)",
            linewidth=line_width,
            alpha=alpha
        )


def plot_training_batches(history_dir, line_width=0.5, alpha=1.0):
    df = pd.read_csv("/".join((history_dir, "history_train_batches.csv")))
    col_names = list(df)
    num_batches = max(df["Batch"])
    x = df["Epoch"] - 1 + (df["Batch"]) / num_batches
    plt.plot(
        x, df["Total Loss"],
        label="Total Loss (training)",
        linewidth=line_width
    )
    # Plot total loss
    for i in range(5, df.shape[1]):
        plt.plot(
            x, df.iloc[:, i],
            label=col_names[i] + " (training)",
            linewidth=line_width,
            alpha=alpha
        )


def plot_validation_history(history_dir, line_width=0.5, alpha=1.0):
    df = pd.read_csv("/".join((history_dir, "history_validation.csv")))
    col_names = list(df)
    # Plot total validation loss
    plt.plot(
        df["Epoch"], df["Total Loss"],
        label="Total Loss (validation)",
        linewidth=line_width
    )
    # Plot validation sub-losses
    for i in range(4, df.shape[1]):
        plt.plot(
            df["Epoch"], df.iloc[:, i],
            label=col_names[i] + " (validation)",
            linewidth=line_width,
            alpha=alpha
        )


if __name__ == "__main__":
    plot_history("/home/samuel/Cranfield University/Code/deeplodocus-dev/test/core/YOLOv3/v04/history",
                 line_width=3,
                 alpha=0.5,
                 scale="linear")
