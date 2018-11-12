import torch.nn as nn
import torch
import os
import __main__

from deeplodocus.utils.flags import *
from deeplodocus.data.dataset import Dataset
from deeplodocus.core.inference.trainer import Trainer
from deeplodocus.core.project.deep_structure.modules.models.classification import Net
from deeplodocus.core.project.deep_structure.modules.metrics.accuracy import accuracy
from deeplodocus.core.metrics.metric import Metric
from deeplodocus.core.metrics.loss import Loss
from deeplodocus.core.inference.tester import Tester
from deeplodocus.utils.notification import Notification
from deeplodocus.utils.logs import Logs
from deeplodocus.core.metrics.over_watch_metric import OverWatchMetric


log_path = os.path.dirname(os.path.abspath(__main__.__file__)) + "/logs"
result_path = os.path.dirname(os.path.abspath(__main__.__file__)) + "/results/history"

logs = [["notification", log_path, ".logs"],
             ["history_train_batches", result_path, ".csv"],
             ["history_train_epochs", result_path, ".csv"],
             ["history_validation", result_path, ".csv"]]

for log_name, log_folder, log_extension in logs:
    Logs(log_name, log_folder, log_extension).check_init()
Notification(DEEP_NOTIF_SUCCESS, "Log and History files initialized ! ", write_logs=False)

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

train_dataset = Dataset(inputs, labels, additional_data, transform_manager=None,  cv_library=DEEP_LIB_PIL, write_logs=False, name="Test Trainer")
train_dataset.load()
train_dataset.set_len_dataset(7)
train_dataset.summary()

# Losses
loss = nn.CrossEntropyLoss()
loss1 = Loss(name="Binary_accuracy1", loss=loss, weight=0.6, write_logs=False)
loss2 = Loss(name="Binary_accuracy2", loss=loss, weight=0.5, write_logs=False)

loss_functions = {loss1.get_name() : loss1, loss2.get_name() :  loss2}


# Metrics
accuracy_metric = Metric(name="Accuracy1", method=loss)
accuracy_metric2 = Metric(name="Accuracy2", method=accuracy)

metrics = {accuracy_metric.get_name() : accuracy_metric, accuracy_metric2.get_name() : accuracy_metric2}

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

tester = Tester(model=model,
                dataset=train_dataset,  # You same dataset to simplify
                metrics={},             # Will be filled by the Trainer
                losses={},              # Will be filled by the Trainer
                batch_size=4,
                num_workers=4,
                verbose=DEEP_VERBOSE_BATCH)




trainer = Trainer(model=model,
                  dataset=train_dataset,
                  losses=loss_functions,
                  metrics=metrics,
                  optimizer=optimizer,
                  num_epochs=10,
                  initial_epoch=1,
                  batch_size=4,
                  shuffle = DEEP_SHUFFLE_ALL,
                  data_to_memorize=DEEP_MEMORIZE_BATCHES,
                  save_condition=DEEP_SAVE_CONDITION_AUTO,
                  verbose=DEEP_VERBOSE_BATCH,
                  num_workers=1,
                  tester=tester,
                  model_name="test-trainer",
                  overwatch_metric= OverWatchMetric(name=TOTAL_LOSS, condition=DEEP_COMPARE_SMALLER),
                  write_logs=False)

trainer.fit()
