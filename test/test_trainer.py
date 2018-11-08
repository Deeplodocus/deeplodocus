import torch.nn.functional as F
import torch.nn as nn
import torch
import collections


from deeplodocus.utils.flags import *
from deeplodocus.data.dataset import Dataset
from deeplodocus.trainer import Trainer
from deeplodocus.core.project.deep_structure.modules.models.classification import Net
from deeplodocus.core.project.deep_structure.modules.metrics.accuracy import accuracy
from deeplodocus.core.metrics.metric import Metric
from deeplodocus.core.metrics.loss import Loss

# Model
model = Net()

# Dataset
inputs = []
labels = []
additional_data = []
inputs.append([r"data/input1.txt"])
inputs.append([r"data/input1.txt"])
labels.append([r"data/label1.txt"])
#inputs.append([r"data/label1.txt"])

train_dataset = Dataset(inputs, labels, additional_data, transform_manager=None,  cv_library=DEEP_LIB_OPENCV, write_logs=False, name="Test Trainer")
train_dataset.load()
train_dataset.set_len_dataset(7)
train_dataset.summary()

# Losses
loss =  nn.CrossEntropyLoss()
loss1 = Loss(name="Binary_accuracy1", loss=loss, weight=0.5, write_logs=False)
loss2 = Loss(name="Binary_accuracy2", loss=loss, weight=0.5, write_logs=False)

loss_functions = {loss1.get_name() : loss1, loss2.get_name() :  loss2}


# Metrics
accuracy_metric = Metric(name="Accuracy1", method=loss)
accuracy_metric2 = Metric(name="Accuracy2", method=accuracy)

metrics = {accuracy_metric.get_name() : accuracy_metric, accuracy_metric2.get_name() : accuracy_metric2}

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


trainer = Trainer(model = model,
                  dataset = train_dataset,
                  losses=loss_functions,
                  metrics=metrics,
                  optimizer=optimizer,
                  num_epochs=10,
                  initial_epoch=1,
                  batch_size=4,
                  shuffle = DEEP_SHUFFLE_ALL,
                  data_to_memorize=DEEP_MEMORIZE_BATCHES,
                  save_condition=DEEP_SAVE_CONDITION_AUTO,
                  num_workers = 1,
                  write_logs=False)

trainer.fit()