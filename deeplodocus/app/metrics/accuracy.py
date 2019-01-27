import torch

def accuracy(output, labels):
    _, output = output.max(1)
    return torch.mean((output == labels).float())
