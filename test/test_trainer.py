import torch.nn.functional as F
import torch

from deeplodocus.utils.types import *
from deeplodocus.data.dataset import Dataset
from deeplodocus.trainer import Trainer
from deeplodocus.core.project.deep_structure.modules.models.LeNet import LeNet
from deeplodocus.core.project.deep_structure.modules.metrics.accuracy import accuracy

# Model
model = LeNet()

# Dataset
inputs = []
labels = []
additional_data = []
inputs.append([r"data/input1.txt"])
labels.append([r"data/label1.txt"])
#inputs.append([r"data/label1.txt"])

train_dataset = Dataset(inputs, labels, additional_data, transform_manager=None,  cv_library=DEEP_OPENCV, write_logs=False, name="Test Trainer")
train_dataset.load()
train_dataset.set_len_dataset(7)
train_dataset.summary()

# Losses
loss_functions = {"Binary_accuracy" : F.nll_loss, "test2" : F.mse_loss}
loss_weights = [0.5, 0.5]

# Metrics
metrics = {"accuracy" : accuracy}

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


trainer = Trainer(model = model,
                  dataset = train_dataset,
                  loss_functions=loss_functions,
                  loss_weights= loss_weights,
                  metrics=metrics,
                  optimizer=optimizer,
                  epochs=10,
                  initial_epoch=0,
                  batch_size=2,
                  shuffle = "all",
                  num_workers = 1,
                  write_logs=False)

trainer.fit()