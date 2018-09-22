
from torch.nn.modules.loss import _Loss

class custom_loss(_Loss):

    def __init__(self):
        super(_Loss)


    def customized_loss(y_estimated, y_true):
        loss = (y_true - y_estimated) ** 2
        return loss

