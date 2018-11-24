#!/usr/bin/env python3

# Back-end imports
from torch import *
import torch
import torch.nn as nn
import torch.nn.functional

# Python imports
from collections import OrderedDict
import numpy as np

from deeplodocus.core.inference.tester import Tester
from deeplodocus.core.inference.trainer import Trainer
from deeplodocus.core.metrics.loss import Loss
from deeplodocus.core.metrics.metric import Metric
from deeplodocus.core.metrics.over_watch_metric import OverWatchMetric
from deeplodocus.core.model.model import Model
from deeplodocus.core.optimizer.optimizer import Optimizer
from deeplodocus.data.dataset import Dataset
from deeplodocus.data.transform_manager import TransformManager
from deeplodocus.utils.dict_utils import check_kwargs
from deeplodocus.utils.dict_utils import get_kwargs
from deeplodocus.utils.flags.lib import *
from deeplodocus.utils.flags.msg import *
from deeplodocus.utils.flags.notif import *
from deeplodocus.utils.flags.path import *
from deeplodocus.utils.flags.dtype import *
from deeplodocus.utils.flags.module import *
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.utils.generic_utils import get_int_or_float
from deeplodocus.utils.notification import Notification


class FrontalLobe(object):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    The FrontalLabe class works as a model manager.
    This class loads :
        - The model
        - The optimizer
        - The trainer
        - The validator
        - The tester

    This class also allows to :
        - Start the training
        - Evaluate the model on the test dataset
        - Display the summaries

    """
    def __init__(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Initialize the Frontal Lobe

        PARAMETERS:
        -----------

        :param config->Namespace: The config

        RETURN:
        -------

        :return: None
        """
        self.config = None
        self.model = None
        self.trainer = None
        self.validator = None
        self.tester = None
        self.metrics = None
        self.losses = None
        self.optimizer = None

    def train(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Start the Trainer

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        self.trainer.fit() if self.trainer is not None else Notification(DEEP_NOTIF_ERROR, DEEP_MSG_NO_TRAINER)

    def evaluate(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Start the Tester

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        self.tester.fit() if self.tester is not None else Notification(DEEP_NOTIF_ERROR, DEEP_MSG_NO_TESTER)

    def load(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the config into the Frontal Lobe

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        self.load_model()        # Always load model first
        self.load_optimizer()    # Always load the optimizer after the model
        self.load_losses()
        self.load_metrics()
        self.load_trainer()
        self.load_validator()
        self.load_tester()
        self.summary()

    def load_model(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Load the model into the Frontal Lobe

        PARAMETERS:
        -----------

        :param config_model -> Namespace: The config of the model

        RETURN:
        -------

        :return model->torch.nn.Module:  The model
        """
        self.model = Model(self.config.model).get()
        Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_MODEL_LOADED %(self.config.model.name, self.model.__module__))

    def load_optimizer(self):
        """
        AUTHORS:
        --------

        :author: Samuel Westlake
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the optimizer with the adequate parameters

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """
        if self.model is not None:
            self.optimizer = Optimizer(self.model.parameters(), self.config.optimizer).get()
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_OPTIMIZER_LOADED %(self.config.optimizer.name, self.optimizer.__module__))
        else:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_OPTIMIZER_NOT_LOADED % DEEP_MSG_MODEL_LOADED)



    def load_losses(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the losses

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return loss_functions->dict: The losses
        """
        losses = {}
        for key, value in self.config.losses.get().items():

            loss = get_module(config=value,
                              modules=DEEP_MODULE_LOSSES)
            kwargs = check_kwargs(get_kwargs(self.config.losses.get(key)))
            method = loss(**kwargs)

            # Check the weight
            if self.config.losses.check("weight", key):
                if get_int_or_float(value.weight) not in (DEEP_TYPE_INTEGER, DEEP_TYPE_FLOAT):
                    Notification(DEEP_NOTIF_FATAL, "The loss function %s doesn't have a correct weight argument" % key)
            else:
                Notification(DEEP_NOTIF_FATAL, "The loss function %s doesn't have any weight argument" % key)

            # Create the loss
            if isinstance(method, torch.nn.Module):
                losses[str(key)] = Loss(name=str(key),
                                        weight=float(value.weight),
                                        loss=method)
                Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_LOSS_LOADED %(key, value.name, loss.__module__))
            else:
                Notification(DEEP_NOTIF_FATAL, "The loss function %s is not a torch.nn.Module instance" % key)
        self.losses = losses

    def load_metrics(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the metrics

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return loss_functions->dict: The metrics
        """
        metrics = {}
        for key, value in self.config.metrics.get().items():
            metric = get_module(config=value,
                               modules=DEEP_MODULE_METRICS)
            kwargs = check_kwargs(get_kwargs(self.config.metrics.get(key)))
            method = metric(**kwargs)

            metrics[str(key)] = Metric(name=str(key), method=method)
            Notification(DEEP_NOTIF_SUCCESS, DEEP_MSG_METRIC_LOADED % (key, value.name, metric.__module__))

        self.metrics = metrics

    def load_trainer(self):
        """
        Author: Alix Leroy and SW
        :return: None
        """
        self.trainer = self.__load_trainer(name="Trainer",
                                           history=self.config.history,
                                           dataloader=self.config.data.dataloader,
                                           data=self.config.data.dataset.train,
                                           transforms=self.config.transform.train)

    def load_validator(self):
        """
        Author: Alix Leroy and SW
        :return: None
        """
        self.validator = self.__load_tester(name="Validator",
                                            dataloader=self.config.data.dataloader,
                                            data=self.config.data.dataset.validation,
                                            transforms=self.config.transform.validation)

    def load_tester(self):
        """
        Author: Alix Leroy and SW
        :return: None
        """
        self.tester = self.__load_tester(name="Tester",
                                         dataloader=self.config.data.dataloader,
                                         data=self.config.data.dataset.test,
                                         transforms=self.config.transform.test)

    def summary(self):
        self.__summary(model=self.model,
                       input_size=self.config.model.input_size,
                       losses=self.losses,
                       metrics=self.metrics,
                       batch_size=self.config.data.dataloader.batch_size)

    def __load_tester(self, dataloader, data, transforms, name):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Load a tester/validator

        PARAMETERS:
        -----------
        :param dataloader:
        :param data:
        :param transforms:
        :param name:

        RETURN:
        -------

        :return tester->Tester: The loaded tester
        """
        inputs = [item for item in data.inputs]
        labels = [item for item in data.labels]
        additional_data = [item for item in data.additional_data]
        transform_manager = TransformManager(transforms)
        dataset = Dataset(list_inputs=inputs,
                          list_labels=labels,
                          list_additional_data=additional_data,
                          transform_manager=transform_manager,
                          cv_library=DEEP_LIB_PIL,
                          name=name)
        dataset.load()
        dataset.set_len_dataset(data.number)
        dataset.summary()
        tester = Tester(model=self.model,
                        dataset=dataset,
                        metrics=self.metrics,
                        losses=self.losses,
                        batch_size=dataloader.batch_size,
                        num_workers=dataloader.num_workers)
        return tester

    def __load_trainer(self, history, dataloader, data, transforms, name):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author: Samuel Westlake

        DESCRIPTION:
        ------------

        Load a trainer

        PARAMETERS:
        -----------
        :param history:
        :param dataloader:
        :param data:
        :param transforms:
        :param name:

        RETURN:
        -------

        :return trainer->Trainer: The loaded trainer
        """
        inputs = [item for item in data.inputs]
        labels = [item for item in data.labels]
        additional_data = [item for item in data.additional_data]
        transform_manager = TransformManager(transforms)
        dataset = Dataset(list_inputs=inputs,
                          list_labels=labels,
                          list_additional_data=additional_data,
                          transform_manager=transform_manager,
                          cv_library=DEEP_LIB_PIL,
                          name=name)
        dataset.load()
        dataset.set_len_dataset(data.number)
        dataset.summary()
        overwatch_metric = OverWatchMetric(name=self.config.training.overwatch_metric,
                                           condition=self.config.training.overwatch_condition)
        trainer = Trainer(model=self.model,
                          dataset=dataset,
                          metrics=self.metrics,
                          losses=self.losses,
                          optimizer=self.optimizer,
                          num_epochs=self.config.training.num_epochs,
                          initial_epoch=self.config.training.initial_epoch,
                          shuffle=self.config.training.shuffle,
                          model_name=self.config.project.name,
                          verbose=history.verbose,
                          tester=self.tester,
                          num_workers=dataloader.num_workers,
                          batch_size=dataloader.batch_size,
                          overwatch_metric=overwatch_metric,
                          save_condition=self.config.training.save_condition,
                          memorize=history.memorize,
                          stopping_parameters=None,
                          history_directory=DEEP_PATH_HISTORY,
                          save_directory=DEEP_PATH_SAVE_MODEL)
        return trainer

    def __summary(self, model, input_size, losses, metrics, batch_size=-1, device="cuda"):
        """
        AUTHORS:
        --------

        :author:  https://github.com/sksq96/pytorch-summary
        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Print a summary of the current model

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return: None
        """

        def register_hook(module):

            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (
                    not isinstance(module, nn.Sequential)
                    and not isinstance(module, nn.ModuleList)
                    and not (module == model)
            ):
                hooks.append(module.register_forward_hook(hook))

        # Check device
        device = device.lower()
        try:
            assert device in ["cuda", "cpu"]
        except AssertionError:
            Notification(DEEP_NOTIF_FATAL, DEEP_MSG_INVALID_DEVICE % device)

        # Set data type depending on device
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # Multiple inputs to the network
        if self.__model_has_multiple_inputs(self.config.data.dataset.train.inputs) is False:
            input_size = [input_size]

        # Batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

        # Create properties
        summary = OrderedDict()
        hooks = []

        # Register hook
        model.apply(register_hook)

        # Make a forward pass
        model(*x)

        # Remove these hooks
        for h in hooks:
            h.remove()

        Notification(DEEP_NOTIF_INFO, '----------------------------------------------------------------')
        line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
        Notification(DEEP_NOTIF_INFO, line_new)
        Notification(DEEP_NOTIF_INFO, '================================================================')
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # Input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']),
                                                      '{0:,}'.format(summary[layer]['nb_params']))
            total_params += summary[layer]['nb_params']
            total_output += np.prod(summary[layer]["output_shape"])
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
            Notification(DEEP_NOTIF_INFO, line_new)

        # Assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        Notification(DEEP_NOTIF_INFO, '================================================================')
        Notification(DEEP_NOTIF_INFO, 'Total params: {0:,}'.format(total_params))
        Notification(DEEP_NOTIF_INFO, 'Trainable params: {0:,}'.format(trainable_params))
        Notification(DEEP_NOTIF_INFO, 'Non-trainable params: {0:,}'.format(total_params - trainable_params))
        Notification(DEEP_NOTIF_INFO, '----------------------------------------------------------------')
        Notification(DEEP_NOTIF_INFO, "Input size (MB): %0.2f" % total_input_size)
        Notification(DEEP_NOTIF_INFO, "Forward/backward pass size (MB): %0.2f" % total_output_size)
        Notification(DEEP_NOTIF_INFO, "Params size (MB): %0.2f" % total_params_size)
        Notification(DEEP_NOTIF_INFO, "Estimated Total Size (MB): %0.2f" % total_size)
        Notification(DEEP_NOTIF_INFO, "----------------------------------------------------------------")

        # List of metrics
        Notification(DEEP_NOTIF_INFO, "LIST OF METRICS :")
        Notification(DEEP_NOTIF_INFO, '================================================================')
        for metric_name, metric in metrics.items():
            Notification(DEEP_NOTIF_INFO, "%s : " % metric_name)

        # List of loss functions
        Notification(DEEP_NOTIF_INFO, "----------------------------------------------------------------")
        Notification(DEEP_NOTIF_INFO, "LIST OF LOSS FUNCTIONS :")
        Notification(DEEP_NOTIF_INFO, '================================================================')
        for loss_name, loss in losses.items():
            Notification(DEEP_NOTIF_INFO, "%s :" % loss_name)

        # Optimizer
        Notification(DEEP_NOTIF_INFO, "----------------------------------------------------------------")
        Notification(DEEP_NOTIF_INFO, "OPTIMIZER :" + str(self.config.optimizer.name))
        Notification(DEEP_NOTIF_INFO, '================================================================')
        for key, value in self.config.optimizer.kwargs.get().items():
            if key != "name":
                Notification(DEEP_NOTIF_INFO, "%s : %s" %(key, value))

    @staticmethod
    def __model_has_multiple_inputs(list_inputs):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the model has multiple inputs

        PARAMETERS:
        -----------

        :param list_inputs(list): The list of inputs in the network

        RETURN:
        -------

        :return->bool: Whether the model has multiple inputs or not
        """
        if len(list_inputs) >= 2:
            return True
        else:
            return False
