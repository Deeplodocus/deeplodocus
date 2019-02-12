import torch
import torch.nn as nn


class DeltaSmaller(nn.Module):

    def __init__(self, p=1):
        super(DeltaSmaller, self).__init__()

        self.p = p

    def forward(self, outputs, targets):
        # Size
        batch_size, channels, height, width = outputs.shape

        return delta_smaller_1_25_p(outputs, targets, p=self.p)/batch_size



def delta_smaller_1_25_p(outputs, labels, p=1):
    v1 = torch.div(outputs, labels)
    v2 = torch.div(labels, outputs)
    delta = torch.max(v1, v2)

    # Compute the number of occurences smaller than teh treshold
    occurences = (delta < 1.25 ** p).sum().type(torch.FloatTensor)

    # Compute teh number of items
    n = float(delta.numel())

    return occurences / n