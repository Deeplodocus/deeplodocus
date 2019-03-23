from decimal import Decimal

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags import TOTAL_LOSS, TRAINING, VALIDATION         # NEEDS TO BE RELOCATED
from deeplodocus.brain.thalamus import Thalamus
from deeplodocus.utils.flags.event import DEEP_EVENT_PRINT_TRAINING_EPOCH_END
from deeplodocus.utils.flags.event import DEEP_EVENT_PRINT_VALIDATION_EPOCH_END
from deeplodocus.utils.flags.event import DEEP_EVENT_PRINT_TRAINING_BATCH_END


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
            expected_arguments=["losses", "total_loss", "metrics", "minibatch_index", "num_minibatches"],
            receiver=self.training_batch_end
        )

    # NB: NOT STATICMETHOD
    def training_batch_end(self, losses, total_loss, metrics, minibatch_index, num_minibatches):
        """
        Print stuff that should be displayed at the end of an epoch
        :return:
        """
        print_metrics = ", ".join(["%s : %.4e" % (TOTAL_LOSS, Decimal(total_loss))]
                                  + ["%s : %.4e" % (loss_name, Decimal(value.item()))
                                     for (loss_name, value) in losses.items()]
                                  + ["%s : %.4e " % (metric_name, Decimal(value))
                                     for (metric_name, value) in metrics.items()])
        Notification(DEEP_NOTIF_RESULT, "[%i/%i] : %s" % (minibatch_index, num_minibatches, print_metrics))

    # NB: NOT STATICMETHOD
    def training_epoch_end(self, losses, total_loss, metrics):
        print_metrics = ", ".join(
            ["%s : %.4e" % (TOTAL_LOSS, Decimal(total_loss))]
            + ["%s : %.4e" % (loss_name, Decimal(value.item()))
               for (loss_name, value) in losses.items()]
            + ["%s : %.4e" % (metric_name, Decimal(value))
               for (metric_name, value) in metrics.items()]
        )
        Notification(DEEP_NOTIF_RESULT, "%s : %s" % (TRAINING, print_metrics))

    # NB: NOT STATICMETHOD
    def validation_epoch_end(self, losses, total_loss, metrics):
        print_metrics = ", ".join(
            ["%s : %.4e" % (TOTAL_LOSS, Decimal(total_loss))]
            + ["%s : %.4e" % (loss_name, Decimal(value.item()))
               for (loss_name, value) in losses.items()]
            + ["%s : %.4e" % (metric_name, Decimal(value))
               for (metric_name, value) in metrics.items()]
        )
        Notification(DEEP_NOTIF_RESULT, "%s: %s" % (VALIDATION, print_metrics))
