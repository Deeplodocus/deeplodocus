import torch


def accuracy(inputs, labels):
    _, inputs = inputs.max(1)
    return torch.mean((inputs == labels).float())


