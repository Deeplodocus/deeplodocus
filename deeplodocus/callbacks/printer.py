from decimal import Decimal

from deeplodocus.utils.notification import Notification
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.flags import *


class Printer(object):

    def __init__(self):
        Thalamus().connect(
            event=DEEP_EVENT_PRINT_TRAINING_EPOCH_END,
            expected_arguments=["epoch_index", "loss", "losses", "metrics"],
            receiver=self.training_epoch_end
        )
        Thalamus().connect(
            event=DEEP_EVENT_PRINT_VALIDATION_EPOCH_END,
            expected_arguments=["epoch_index", "loss", "losses", "metrics"],
            receiver=self.validation_epoch_end
        )
        Thalamus().connect(
            event=DEEP_EVENT_PRINT_TRAINING_BATCH_END,
            expected_arguments=["batch_index", "num_batches", "epoch_index", "loss", "losses", "metrics"],
            receiver=self.training_batch_end
        )

    def training_batch_end(self, loss, losses, metrics, epoch_index, batch_index, num_batches):
        """
        Print stuff that should be displayed at the end of an epoch
        :return:
        """
        Notification(
            DEEP_NOTIF_RESULT, "[%i : %i/%i] : %s" % (
                epoch_index,
                batch_index,
                num_batches,
                self.compose_text(loss, losses, metrics)
            )
        )

    def training_epoch_end(self, epoch_index, loss, losses, metrics):
        Notification(
            DEEP_NOTIF_RESULT,
            "Epoch %i : %s : %s" % (epoch_index, TRAINING, self.compose_text(loss, losses, metrics))
        )

    def validation_epoch_end(self, epoch_index, loss, losses, metrics):
        Notification(
            DEEP_NOTIF_RESULT,
            "Epoch %i : %s : %s" % (epoch_index, VALIDATION, self.compose_text(loss, losses, metrics))
        )

    @staticmethod
    def compose_text(total_loss, losses, metrics, sep=" : "):
        return sep.join(
            ["%s : %.4e" % (DEEP_LOG_TOTAL_LOSS.name, Decimal(total_loss))]
            + ["%s : %.4e" % (loss_name, Decimal(value)) for (loss_name, value) in losses.items()]
            + ["%s : %.4e " % (metric_name, Decimal(value)) for (metric_name, value) in metrics.items()]
        )
