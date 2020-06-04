from typing import Union
from torch.utils.data import DataLoader
from math import ceil

from deeplodocus.core.metrics import Losses, Metrics
from deeplodocus.data.load.dataset import Dataset
from deeplodocus.flags import *
from deeplodocus.utils.generic_utils import get_corresponding_flag
from deeplodocus.utils.namespace import Namespace


class Inferer(object):

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
            name: str = "Inferer"
    ):
        self.dataset = dataset
        self.model = model
        self.transform_manager = transform_manager
        self.losses = losses
        self.metrics = Metrics() if metrics is None else metrics
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.name = name
        self.shuffle = get_corresponding_flag(
            DEEP_LIST_SHUFFLE, shuffle,
            fatal=False,
            default=DEEP_SHUFFLE_NONE
        )
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def get_num_batches(self) -> int:
        return int(ceil(len(self.dataset) / self.batch_size))

    def to_device(self, x, device):
        if isinstance(x, list):
            return [self.to_device(i, device) for i in x]
        elif isinstance(x, tuple):
            return tuple([self.to_device(i, device) for i in x])
        elif isinstance(x, dict):
            return {k: self.to_device(i, device) for k, i in x.items()}
        elif isinstance(x, Namespace):
            x.__dict__ = {k: self.to_device(i, device) for k, i in x.__dict__.items()}
            return x
        else:
            try:
                return x.to(device)
            except AttributeError:
                return x

    def detach(self, x):
        if isinstance(x, list):
            return [self.detach(item) for item in x]
        elif isinstance(x, tuple):
            return tuple([self.detach(item) for item in x])
        elif isinstance(x, dict):
            return {key: self.detach(item) for key, item in x.items()}
        elif isinstance(x, Namespace):
            x.__dict__ = {k: self.detach(i) for k, i in x.__dict__.items()}
            return x
        else:
            try:
                return x.detach()
            except AttributeError:
                return x

    @staticmethod
    def clean_single_element_list(batch: list) -> list:
        cleaned_minibatch = []
        # For each entry in the minibatch:
        # If it is a single element list -> Make it the single element
        # If it is an empty list -> Make it None
        # Else -> Do not change
        for item in batch:
            if isinstance(item, list) and len(item) == 1:
                cleaned_minibatch.append(item[0])
            elif isinstance(item, list) and len(item) == 0:
                cleaned_minibatch.append(None)
            else:
                cleaned_minibatch.append(item)
        return cleaned_minibatch

    @staticmethod
    def compose_text(loss: float, losses: dict, metrics: dict, sep: str = " : "):
        return sep.join(
            ["%s : %.4e" % (DEEP_LOG_TOTAL_LOSS.name, loss)]
            + ["%s : %.4e" % (loss_name, value) for (loss_name, value) in losses.items()]
            + ["%s : %.4e " % (metric_name, value) for (metric_name, value) in metrics.items()]
        )