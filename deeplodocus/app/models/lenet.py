import torch.nn as nn
import torch.nn.functional as F

# LeNet Model definition


class LeNet(nn.Module):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Implementation of LeNet as defined in : http://yann.lecun.com/exdb/lenet/
    The implementation is slightly modified with the number of hidden units in the fully connected layers
    """
    def __init__(self, num_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

