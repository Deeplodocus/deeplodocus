from typing import Union

from deeplodocus.core.metrics import Losses, Metrics
from deeplodocus.data.load.dataset import Dataset
from deeplodocus.flags import *
from deeplodocus.utils.generic_utils import ProgressBar
from deeplodocus.utils.notification import Notification
from deeplodocus.core.inference import Inferer


class Tester(Inferer):

    def __init__(
            self,
            dataset: Dataset,
            model,
            transform_manager,
            losses: Losses,
            metrics: Union[Metrics, None] = None,
            batch_size: int = 32,
            num_workers: int = 1,
            shuffle: Flag = DEEP_SHUFFLE_NONE,
            name: str = "Tester"
    ):
        super(Tester, self).__init__(
            dataset, model, transform_manager, losses,
            metrics=metrics,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            name=name
        )
        self.progress_bar = None

    def evaluate(self, silent: bool = False, progress_bar: Union[ProgressBar, bool] = True, prefix: str = "Evaluation :"):
        self.evaluation_start(silent=silent, progress_bar=progress_bar, prefix="DEEP PROGRESS : %s" % prefix)
        for batch in self.dataloader:
            self.evaluation_batch(batch)
        return self.evaluation_end(silent=silent)

    def evaluation_start(self, silent: bool = False, progress_bar: bool = False, prefix: str = "Evaluation :"):
        if progress_bar is True:
            self.progress_bar = ProgressBar(self.get_num_batches(), prefix=prefix)
        elif isinstance(progress_bar, ProgressBar):
            self.progress_bar = progress_bar
        if not silent:
            Notification(DEEP_NOTIF_INFO, DEEP_MSG_EVALUATION_STARTED)
        self.model.eval()  # Put model into evaluation mode
        self.losses.reset(self.dataset.type)  # Reset corresponding losses
        self.metrics.reset(self.dataset.type)  # Reset corresponding metrics

    def evaluation_end(self, silent: bool = False):
        self.transform_manager.finish()  # Call finish on all output transforms
        loss, losses = self.losses.reduce(self.dataset.type)  # Get total loss and mean of each loss
        metrics = self.metrics.reduce(self.dataset.type)  # Get total metric values
        if not silent:
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_EVALUATION_FINISHED)
            Notification(DEEP_NOTIF_RESULT, self.compose_text(loss, losses, metrics))
        return loss, losses, metrics

    def evaluation_batch(self, batch):
        inputs, labels, additional_data = self.clean_single_element_list(batch)
        inputs = self.to_device(inputs, self.model.device)  # Send data to device
        labels = self.to_device(labels, self.model.device)
        additional_data = self.to_device(labels, self.model.device)
        with torch.no_grad():
            outputs = self.model(*inputs)  # Infer the outputs from the model over the given mini batch
        outputs = self.detach(outputs)  # Detach the tensor from the graph
        self.losses.forward(self.dataset.type, outputs, labels, inputs, additional_data)  # Compute losses
        outputs = self.transform_manager.transform(outputs, inputs, labels, additional_data)  # Output transforms
        self.metrics.forward(self.dataset.type, outputs, labels, inputs, additional_data)  # Compute metrics
        if self.progress_bar is not None:
            self.progress_bar.step()