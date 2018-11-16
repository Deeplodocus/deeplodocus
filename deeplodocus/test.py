import torch


from deeplodocus.core.optimizer.optimizer import Optimizer
from deeplodocus.core.project.deep_structure.modules.models.classification import Net

net = Net()
print(net.parameters())
Optimizer("sgd", write_logs=False, params=net.parameters(), lr=1)