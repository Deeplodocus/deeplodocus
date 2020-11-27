from deeplodocus.utils.notification import Notification
from deeplodocus.flags import TOTAL_LOSS, TRAINING, VALIDATION         # NEEDS TO BE RELOCATED
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.flags.event import DEEP_EVENT_PRINT_TRAINING_EPOCH_END
from deeplodocus.flags.event import DEEP_EVENT_PRINT_VALIDATION_EPOCH_END
from deeplodocus.flags.event import DEEP_EVENT_PRINT_TRAINING_BATCH_END
from deeplodocus.flags.notif import *


class Printer(object):

    def __init__(self):
        Thalamus().connect(
            event=DEEP_EVENT_PRINT_TRAINING_EPOCH_END,
            expected_arguments=["losses", "total_loss", "metrics"],
            receiver=self.training_epoch_end
        )
        Thalamus().connect(
            event=DEEP_EVENT_PRINT_VALIDATION_EPOCH_END,
            expected_arguments=["losses", "total_loss", "metrics"],
            receiver=self.validation_epoch_end
        )
        Thalamus().connect(
            event=DEEP_EVENT_PRINT_TRAINING_BATCH_END,
            expected_arguments=[
                "losses",
                "total_loss",
                "metrics",
                "minibatch_index",
                "num_minibatches",
                "epoch_index"
            ],
            receiver=self.training_batch_end
        )

    # NB: NOT STATICMETHOD
    def training_batch_end(self, losses, total_loss, metrics, epoch_index, minibatch_index, num_minibatches):
        """
        Print stuff that should be displayed at the end of an epoch
        :return:
        """
        print_metrics = ", ".join(
            ["%s : %.4e" % (TOTAL_LOSS, total_loss)]
            + ["%s : %.4e" % (loss_name, value) for (loss_name, value) in losses.items()]
            + ["%s : %.4e " % (metric_name, value) for (metric_name, value) in metrics.items()]
        )
        Notification(
            DEEP_NOTIF_RESULT,
            "[%i : %i/%i] : %s" % (epoch_index, minibatch_index, num_minibatches, print_metrics)
        )

    # NB: NOT STATICMETHOD
    def training_epoch_end(self, losses, total_loss, metrics):
        print_metrics = ", ".join(
            ["%s : %.4e" % (TOTAL_LOSS, total_loss)]
            + ["%s : %.4e" % (loss_name, value) for (loss_name, value) in losses.items()]
            + ["%s : %.4e" % (metric_name, value) for (metric_name, value) in metrics.items()]
        )
        Notification(DEEP_NOTIF_RESULT, "%s : %s" % (TRAINING, print_metrics))

    # NB: NOT STATICMETHOD
    def validation_epoch_end(self, losses, total_loss, metrics):
        print_metrics = ", ".join(
            ["%s : %.4e" % (TOTAL_LOSS, total_loss)]
            + ["%s : %.4e" % (loss_name, value) for (loss_name, value) in losses.items()]
            + ["%s : %.4e" % (metric_name, value) for (metric_name, value) in metrics.items()]
        )
        Notification(DEEP_NOTIF_RESULT, "%s: %s" % (VALIDATION, print_metrics))
