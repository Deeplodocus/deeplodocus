#!/usr/bin/env python3

from torch import *
import torch
import torch.nn as nn

from collections import OrderedDict

import copy
import shutil
import numpy as np

import matplotlib.pyplot as plt

from deeplodocus.utils.notification import Notification
from deeplodocus.utils.flags import *
from deeplodocus.core.metrics.loss import Loss
from deeplodocus.core.metrics.metric import Metric
from deeplodocus.core.inference.tester import Tester
from deeplodocus.data.dataset import Dataset
from deeplodocus.data.transform_manager import TransformManager
from deeplodocus.core.optimizer.optimizer import Optimizer

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
    def __init__(self, config, write_logs: bool = True):
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
        self.config = config
        self.model = None
        self.trainer = None
        self.validator = None
        self.tester = None
        self.metrics = None
        self.losses = None
        self.optimizer = None
        self.write_logs=write_logs
        # Load the attributes

    def train(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

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

        if self.trainer is not None:
            self.trainer.fit()
        else:
            Notification(DEEP_NOTIF_ERROR, "The trainer is not loaded, please make sure all the config files are correct", write_logs=self.write_logs)

    def evaluate(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

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

        if self.tester is not None:
            self.tester.fit()


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

        # Model (always before the optimizer)
        self.model = self.__load_model()

        # Optimizer (always after the Model)
        self.optimizer = self.__load_optimizer()

        # Losses
        self.losses = self.__load_losses()

        # Metrics
        self.metrics = self.__load_metrics()
        """
        self.validator = self.__load_tester(name="Validator",
                                            dataloader=self.config.data.dataloader,
                                            dataset=self.config.data.dataset.validation,
                                            transforms=self.config.transform.validation)

        Notification(DEEP_NOTIF_SUCCESS, "Validator loaded", write_logs=self.write_logs)

        self.tester = self.__load_tester(name="Tester",
                                         dataloader=self.config.data.dataloader,
                                         dataset=self.config.data.dataset.test,
                                         transforms=self.config.transform.test)

        Notification(DEEP_NOTIF_SUCCESS, "Tester loaded", write_logs=self.write_logs)

        #self.trainer = self.__load_trainer(, tester = self.validator)

        Notification(DEEP_NOTIF_SUCCESS, "Trainer loaded", write_logs=self.write_logs)
        """

        self.__summary(model=self.model,
                       input_size=self.config.model.input_size,
                       losses=self.losses, metrics = self.metrics,
                       batch_size=self.config.data.dataloader.batch_size)


    def __load_optimizer(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the optimizer

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return optimizer->torch.nn.Module:
        """

        return Optimizer(params=self.model.parameters(), write_logs=self.write_logs, **self.config.optimizer.get()).get()



    def __load_losses(self):
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
        loss_functions= {}

        for key, value in self.config.losses.get().items():

            # Get the loss method
            local = {"method": None}
            try:
                exec("from {0} import {1} \nmethod= {2}".format(value.path, value.method, value.method), {}, local)
            except:
                Notification(DEEP_NOTIF_ERROR,
                             DEEP_MSG_LOSS_NOT_FOUND % (value.method),
                             write_logs=self.write_logs)

            if self.config.losses.check("kwargs", key):
                method = local["method"](value.kwargs)
            else:
                method = local["method"]()

            # Check if the loss is custom
            if value.path == "torch.nn":
                is_custom = False
            else:
                is_custom = True

            if isinstance(method, torch.nn.Module):
                loss_functions[str(key)] = Loss(name=str(key), is_custom=is_custom, weight=float(value.weight), loss=method, write_logs=self.write_logs)
            else:
                Notification(DEEP_NOTIF_FATAL, "The loss function %s is not a torch.nn.Module instance" %str(key), write_logs=self.write_logs)

        return loss_functions


    def __load_metrics(self):
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

        :return loss_functions->dict: The losses
        """
        metric_functions= {}

        for key, value in self.config.metrics.get().items():

            # Get the metric method
            local = {"method": None}
            try:
                exec("from {0} import {1} \nmethod= {2}".format(value.path, value.method, value.method), {}, local)
            except:
                Notification(DEEP_NOTIF_ERROR,
                             DEEP_MSG_METRIC_NOT_FOUND % (value.method),
                             write_logs=self.write_logs)

            if self.config.metrics.check("kwargs", key):
                method = local["method"](value.kwargs)
            else:
                method = local["method"]()

            metric_functions[str(key)] = Metric(name=str(key), method=method, write_logs=self.write_logs)

        return metric_functions

    def __load_model(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load the model into the Frontal Lobe

        PARAMETERS:
        -----------

        :param config_model->Namespace: The config of the model

        RETURN:
        -------

        :return model->torch.nn.Module:  The model
        """

        local = {"model" : None}
        try:
            exec("from {0} import {1} \nmodel= {2}".format(self.config.model.module, self.config.model.name, self.config.model.name), {}, local)
        except:
            Notification(DEEP_NOTIF_ERROR, DEEP_MSG_MODEL_NOT_FOUND %(self.config.model.name, self.config.model.module), write_logs=self.write_logs)


        if self.config.check("kwargs", "model"):
            model = local["model"](self.config.model.kwarg)
        else:
            model = local["model"]()

        return model


    def __load_tester(self, dataloader, dataset, transforms, name):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Load a tester/validator

        PARAMETERS:
        -----------
        :param dataloader:
        :param dataset:
        :param transforms:

        RETURN:
        -------

        :return tester->Tester: The loaded tester
        """

        # Dataset
        inputs = []
        labels = []
        additional_data = []

        for input in dataset.inputs:
            inputs.append(input)

        for label in dataset.labels:
            labels.append(label)

        for add_data in dataset.additional_data:
            additional_data.append(add_data)


        # Create the transform managers
        transform_manager = TransformManager(transforms, write_logs=False)

        dataset = Dataset(list_inputs=inputs,
                          list_labels=labels,
                          list_additional_data=additional_data,
                          transform_manager=transform_manager,
                          cv_library=DEEP_LIB_PIL,
                          write_logs=self.write_logs,
                          name=name)

        dataset.load()
        dataset.summary()

        tester = Tester(model = self.model,
                        dataset=dataset,
                        metrics=self.metrics,
                        losses=self.losses,
                        batch_size=dataloader.batch_size,
                        num_workers=dataloader.num_workers)

        return tester

    def __summary(self, model, input_size, losses, metrics, batch_size=-1, device="cuda"):
        """
        AUTHORS:
        --------

        :author: Alix Leroy
        :author:  https://github.com/sksq96/pytorch-summary

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
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
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

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # multiple inputs to the network
        if self.__model_has_multiple_inputs() is False:
            input_size = [input_size]



        print(input_size)
        # batch_size of 2 for batchnorm
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

        # print(type(x[0]))
        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        model(*x)
        # remove these hooks
        for h in hooks:
            h.remove()

        Notification(DEEP_NOTIF_INFO, '----------------------------------------------------------------', write_logs=self.write_logs)
        line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
        Notification(DEEP_NOTIF_INFO, line_new, write_logs=self.write_logs)
        Notification(DEEP_NOTIF_INFO, '================================================================', write_logs=self.write_logs)
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']),
                                                      '{0:,}'.format(summary[layer]['nb_params']))
            total_params += summary[layer]['nb_params']
            total_output += np.prod(summary[layer]["output_shape"])
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
            Notification(DEEP_NOTIF_INFO, line_new, write_logs=self.write_logs)

        # assume 4 bytes/number (float on cuda).
        total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
        total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
        total_size = total_params_size + total_output_size + total_input_size

        Notification(DEEP_NOTIF_INFO, '================================================================', write_logs=self.write_logs)
        Notification(DEEP_NOTIF_INFO, 'Total params: {0:,}'.format(total_params), write_logs=self.write_logs)
        Notification(DEEP_NOTIF_INFO, 'Trainable params: {0:,}'.format(trainable_params), write_logs=self.write_logs)
        Notification(DEEP_NOTIF_INFO, 'Non-trainable params: {0:,}'.format(total_params - trainable_params), write_logs=self.write_logs)
        Notification(DEEP_NOTIF_INFO, '----------------------------------------------------------------', write_logs=self.write_logs)
        Notification(DEEP_NOTIF_INFO, "Input size (MB): %0.2f" % total_input_size, write_logs=self.write_logs)
        Notification(DEEP_NOTIF_INFO, "Forward/backward pass size (MB): %0.2f" % total_output_size, write_logs=self.write_logs)
        Notification(DEEP_NOTIF_INFO, "Params size (MB): %0.2f" % total_params_size, write_logs=self.write_logs)
        Notification(DEEP_NOTIF_INFO, "Estimated Total Size (MB): %0.2f" % total_size, write_logs=self.write_logs)
        Notification(DEEP_NOTIF_INFO, "----------------------------------------------------------------", write_logs=self.write_logs)

        # List of metrics
        Notification(DEEP_NOTIF_INFO, "LIST OF METRICS :", write_logs=self.write_logs)
        for metric_name, metric in metrics.items():
            Notification(DEEP_NOTIF_INFO, "%s :" % metric_name, write_logs=self.write_logs)

        # List of loss functions
        Notification(DEEP_NOTIF_INFO, "LIST OF LOSS FUNCTIONS :", write_logs=self.write_logs)
        for loss_name, loss in losses.items():
            Notification(DEEP_NOTIF_INFO, "%s :" % loss_name, write_logs=self.write_logs)

        # Optimizer
        Notification(DEEP_NOTIF_INFO, "OPTIMIZER :" + str(self.config.optimizer.name), write_logs=self.write_logs)
        for key, value in self.config.optimizer.get().items():
            if key != "name":
                Notification(DEEP_NOTIF_INFO, "%s : %s" %(key, value), write_logs=self.write_logs)


    def __model_has_multiple_inputs(self):
        """
        AUTHORS:
        --------

        :author: Alix Leroy

        DESCRIPTION:
        ------------

        Check if the model has multiple inputs

        PARAMETERS:
        -----------

        None

        RETURN:
        -------

        :return->bool: Whether the model has multiple inputs or not
        """

        if len(self.config.data.dataset.train.inputs) >= 2:
            return True
        else:
            return False


    def patchy(self):
        """
        Author: Samuel Westlake
        This method slides a patch over the image and makes a prediction for each patch position
        The 1 - prediction score for the given ground truth is saved in an output matrix
        :return:
        """
        # Get settings from config
        shape = self.config["data_gen"]["common"]["x_shape"]
        show = self.config["patchyt"]["show"]
        scale = self.config["patchy"]["scale"]
        bias = self.config["patchy"]["bias"]
        keep_aspect = self.config["patchy"]["keep_aspect"]
        path = self.config["patchy"]["path"]
        patch_size = self.config["patchy"]["patch_size"]
        ground_truth = self.config["patchy"]["ground_truth"]
        directory = "%s/%s" % (self.config["working_dir"], self.config["name"])

        # If path is an image_path, make it a list, if path is a dir_path, get all files
        if not os.path.isfile(path):
            print("Error: %s not found" % path)
            return

        image = cv2.imread(path)
        image = image_aug.resize(image, shape, keep_aspect).astype(np.float32)
        image = image * scale + bias
        image_name = path.split("/")[-1].split(".")[0]

        # Initialise output, batches and indexes
        output = np.empty((shape[0], shape[1]), dtype=np.float32)
        batch = np.empty([shape[1]] + list(shape), dtype=np.float32)
        for j in range(shape[0]):
            for i in range(shape[1]):
                patched = self.__apply_patch(copy.deepcopy(image), (i, j), patch_size)
                batch[i] = patched
                if show:
                    cv2.imshow("patch", patched)
                    cv2.waitKey(1)
            y_hat = self.net.predict_on_batch(batch)
            for i, prediction in enumerate(y_hat):
                output[j][i] = 1 - prediction[ground_truth]
        if show:
            cv2.destroyWindow("patch")

        # Plot graphs
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap="seismic")
        plt.title("Input")
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.imshow(output, cmap="seismic")
        plt.title("Prediction")
        plt.grid()
        plt.savefig("%s/%s_patch.png" % (directory, image_name))
        print("Plot saved to %s/%s_patchy.png" % (directory, image_name))







    @staticmethod
    def __set_fit_class_weight(fit_gen_kwargs, generator):
        """
        Compute class weights before training using method specified
        :param fit_gen_kwargs: the dictionary of key word arguments for training
        :param generator: the date generator to be used
        :return:
        """
        class_weight = fit_gen_kwargs["class_weight"]
        if isinstance(class_weight, str):
            fit_gen_kwargs["class_weight"] = generator.class_weights(class_weight)
        return fit_gen_kwargs

    @staticmethod
    def __build_from_list(architecture):
        model = Sequential()
        layer_names = set([layer_name for layer in architecture for layer_name in layer])
        for layer_name in layer_names:
            exec("from tensorflow.python.keras.layers import %s" % layer_name)
        for layer in architecture:
            for layer_name, kwargs in layer.items():
                kwargs = kwargs if kwargs is not None else {}
                exec("model.add(%s(**%s))" % (layer_name, kwargs))
        return model

    @staticmethod
    def __build_from_module(module_name, args, kwargs):
            exec("import sys")
            exec('sys.path.append("..")')                                       # Adds higher directory to python modules path.
            exec("from modules.models import %s" % module_name)
            exec("model = %s.build(**%s)" % (module_name, kwargs))
            return locals()["model"]

    @staticmethod
    def __get_args_and_kwargs(params):
        """
        Returns a list and dict of args and kwargs from a given dict
        :param params: a dict
        :return: args, a list of arguments; kwargs, a dict or keyword arguments
        """
        args, kwargs = [], {}
        if params is not None:
            if "args" in params:
                args = params["args"]
            if "kwargs" in params:
                kwargs = params["kwargs"]
        return args, kwargs


    @staticmethod
    def __apply_patch(image, pos, size):
        i, j = pos
        shape = image.shape
        lj = int(max(0, int(j - size / 2)))
        uj = int(min(np.ceil(j + size / 2), shape[0]))
        li = int(max(0, int(i - size / 2)))
        ui = int(min(np.ceil(i + size / 2), shape[0]))
        image[lj:uj, li:ui, :] = 0
        return image

    @staticmethod
    def __copy_incorrect(src, destination, ground_truth, prediction):
        """
        Copies an image in src destination/ground_truth/prediction
        :param src: string, path to image
        :param destination: string, path to destination directory
        :param ground_truth: int, ground truth of image
        :param prediction: int, predicted class of image
        :return: None
        """
        image_name = src.split("/")[-1]
        new_dir = "/".join([destination, str(ground_truth), str(prediction)])
        os.makedirs(new_dir, exist_ok=True)
        shutil.copyfile(src, "/".join([new_dir, image_name]))


