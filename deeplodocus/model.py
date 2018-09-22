#!/usr/bin/env python3


import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict

import os
import yaml
import cv2
import copy
import shutil
import numpy as np
from .trainer import Trainer
from .data_generator import DataGenerator
from .callbacks.history import History
from .callbacks.model_checkpoint import ModelCheckpoint
from .preprocessing import image_aug
import matplotlib.pyplot as plt

from .helpers import helpers
from .dataset import Dataset


class Model(object):

    def __init__(self, config_path):
        """
        :param config_path: String. Path to config file.
        """
        self.config_path = config_path
        self.config = {}
        self.net = None
        self.history = None
        self.evaluation = {}
        self.load_config()
        self.__on_start()

    def load_config(self, new_values=None):
        """
        Author: Samuel Westlake and Alix Leroy
        Loads model configurations from the yaml file specified by self.config_path into self.config
        If new_values are specified, they are then added to self.config
        self.history is re-initialized for the new config
        :return:
        """
        self.config = helpers.load_yaml(self.config_path)
        if new_values is not None:
            self.__config_update(new_values)
        self.__initialize_history()

    def config_add(self, new_values):
        """
        Changes self.config with the values from a given dictionary
        self.history is re-initialized for the new config
        :param new_values:
        :return:
        """
        self.__config_update(new_values)
        self.__initialize_history()

    def show_config(self, sub_dict=None):
        """
        Prints the config to screen
        If a sub_dict is given (level 1 only) then only that sub_dict is printed
        :return:
        """
        config = self.config
        if sub_dict is not None:
            config = config[sub_dict]
        if isinstance(config, dict):
            helpers.print_dict(config)
        else:
            print(config)

    def save_config(self, file_path=None, to_working_dir=False):
        """
        Author: Samuel Westlake
        Save the config file to output_dir/name.yaml
        If timestamp is not None, the timestamp is included
        :return: None
        """
        if file_path is None:
            directory = "%s/%s" % (self.config["working_dir"], self.config["name"])
            os.makedirs(directory, exist_ok=True)
            file_name = "%s_config_%s.yaml" % (self.config["name"], self.history.epoch)
            file_path = "%s/%s" % (directory, file_name)
        else:
            if to_working_dir:
                file_path = "%s/%s/%s" % (self.config["working_dir"], self.config["name"], file_path)
        with open(file_path, "w") as yaml_file:
            yaml_file.write(yaml.dump(self.config, default_flow_style=False))

    def get_config(self):
        """
        :return: self.config
        """
        return self.config

    def show_history(self):
        print(self.history.history)


    def summary(self, model, input_size, losses, metrics, optimizer, device="cuda",):
        """
        Author :Alix Leroy
        Inspired from : https://github.com/sksq96/pytorch-summary
        Print a summary of the current model
        :return: None
        """
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split('.')[-1].split("'")[0]
                module_idx = len(summary)

                m_key = '%s-%i' % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]['input_shape'] = list(input[0].size())
                summary[m_key]['input_shape'][0] = -1
                if isinstance(output, (list, tuple)):
                    summary[m_key]['output_shape'] = [[-1] + list(o.size())[1:] for o in output]
                else:
                    summary[m_key]['output_shape'] = list(output.size())
                    summary[m_key]['output_shape'][0] = -1

                params = 0
                if hasattr(module, 'weight') and hasattr(module.weight, 'size'):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]['trainable'] = module.weight.requires_grad
                if hasattr(module, 'bias') and hasattr(module.bias, 'size'):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]['nb_params'] = params

            if (not isinstance(module, nn.Sequential) and
                    not isinstance(module, nn.ModuleList) and
                    not (module == model)):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in ["cuda", "cpu"], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        # check if there are multiple inputs to the network
        if isinstance(input_size[0], (list, tuple)):
            x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
        else:
            x = Variable(torch.rand(2, *input_size)).type(dtype)

        # print(type(x[0]))
        # create properties
        summary = OrderedDict()
        hooks = []
        # register hook
        model.apply(register_hook)
        # make a forward pass
        # print(x.shape)
        model(x)
        # remove these hooks
        for h in hooks:
            h.remove()

        print('----------------------------------------------------------------')
        line_new = '{:>20}  {:>25} {:>15}'.format('Layer (type)', 'Output Shape', 'Param #')
        print(line_new)
        print('================================================================')
        total_params = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = '{:>20}  {:>25} {:>15}'.format(layer, str(summary[layer]['output_shape']),
                                                      '{0:,}'.format(summary[layer]['nb_params']))
            total_params += summary[layer]['nb_params']
            if 'trainable' in summary[layer]:
                if summary[layer]['trainable'] == True:
                    trainable_params += summary[layer]['nb_params']
            print(line_new)
        print('================================================================')
        print('Total params: {0:,}'.format(total_params))
        print('Trainable params: {0:,}'.format(trainable_params))
        print('Non-trainable params: {0:,}'.format(total_params - trainable_params))
        print('----------------------------------------------------------------')

        print("=================================================================")
        print("LIST OF METRICS :")

        for metric in metrics:
            print("\t" + str(metric[0]))
        print("------------------------------------------------------------------")

        print("=================================================================")
        print("LIST OF LOSS FUNCTIONS:")

        for loss in losses:
            print("\t" + str(loss[0]))
        print("------------------------------------------------------------------")

        print("=================================================================")
        print("OPTIMIZER :" + str(optimizer))



    # return summary

    def build(self):
        """
        Author : Samuel Westlake and Alix Leroy
        Build the model
        :return: None
        """
        # Extract the build method from config
        method = self.config["build"]["method"]

        # Set the model using the appropriate build method
        if method == "yaml_path":
            yaml_name = self.config["build"]["yaml_path"]
            self.net = self.__build_from_list(helpers.load_yaml("%s" % yaml_name))

        elif method == "architecture":
            self.net = self.__build_from_list(self.config["build"]["architecture"])

        elif method == "module":
            module_name = self.config["build"]["module"]["name"]
            args, kwargs = self.__get_args_and_kwargs(self.config["build"]["module"])
            self.net = self.__build_from_module(module_name, args, kwargs)

        else:
            raise ValueError("The BUILD method selected in the config file does not exist. \n" 
                             "Please select one of the following methods : \n"
                             "\t - yaml_path \n"
                             "\t - architecture \n"
                             "\t - module")


        self.optimizer = self.__get_optimizer()                                              # Select the optimizer

        self.losses = self.__get_losses()                                                        # Get the loss functions

        self.metrics = self.__get_metrics()

        self.input_size = self.__get_input_size()
        self.device = self.config["build"]["device"]

        #Print the model summary
        self.summary(input_size=self.input_size, device=self.device, losses=self.losses, metrics=self.metrics, optimizer=self.optimizer)

    def train(self):
        """
        Author: Alix Leroy
        Call the appropriate training method
        :return: None
        """
        method = self.config["data_gen"]["common"]["method"]
        self.save_config()


        trainer = Trainer(criterion=criterion, optimizer=self.optimizer, metrics=self.metrics, losses=self.losses)

        trainer.start(method)




    def load_model(self, file_path=None, from_working_dir=False):
        """
        Author : Samuel Westlake and Alis Leroy
        Load an external model into the model object
        :return: None
        """
        if file_path is None:
            if self.config["load"]["from_working_dir"]:
                file_path = "%s/%s/%s" % (self.config["working_dir"], self.config["name"], self.config["load"]["path"])
            else:
                file_path = self.config["load"]["path"]
        else:
            if from_working_dir:
                file_path = "%s/%s/%s" % (self.config["load"]["from_working_dir"], self.config["name"], file_path)

        self.net = load_model(file_path)
        self.__initialize_history()
        try:
            self.history.epoch = int(file_path.split("/")[-1].split("_")[-2])
            print("Current epoch set to %i" % self.history.epoch)
        except ValueError:
            print("Current epoch set to 0")

    def load_weights(self, file_path=None, from_working_dir=False):
        """
        Author : Samuel Westlake and Alix Leroy
        Load an external model into the model object
        :return: None
        """
        if file_path is None:
            if self.config["load"]["from_working_dir"]:
                file_path = "%s/%s/%s" % (self.config["working_dir"], self.config["name"], self.config["load"]["path"])
            else:
                file_path = self.config["load"]["path"]
        else:
            if from_working_dir:
                file_path = "%s/%s/%s" % (self.config["load"]["from_working_dir"], self.config["name"], file_path)

        self.net.load_weights(file_path)
        self.__initialize_history()
        try:
            self.history.epoch = int(file_path.split("/")[-1].split("_")[-2])
            print("Current epoch set to %s", self.history.epoch)
        except:
            print("Current epoch set to 0")

    def save_model(self, file_path=None, to_working_dir=False):
        """
        Author: Samuel and Alix Leroy
        Save the model to a file path given by the user
        :return: None
        """
        history = self.history.history
        epoch = self.history.epoch
        if file_path is None:
            metric = self.config["callbacks"]["checkpoint"]["args"]["monitor"]
            directory = "%s/%s" % (self.config["directory"], self.config["name"])
            if not history.empty:
                metric = history[metric][history.index[epoch - 1]]
            else:
                metric = 0
            name = "%s_model_%i_%.2f.h5" % (self.config["name"], epoch, metric)
            file_path = "%s/%s" % (directory, name)
        else:
            if to_working_dir:
                file_path = "%s/%s/%s" % (self.config["working_dir"], self.config["name"], file_path)

        if len(file_path.split("/")) > 1:                               # If the path contains a directory
            directory = "/".join(file_path.split("/")[:-1])             # Store the directory path
            os.makedirs(directory, exist_ok=True)                       # Make the directory if necessary
        self.net.save(file_path)                                      # Save model
        print("Model saved to %s" % file_path)

    def save_weights(self, file_path=None, to_working_dir=False):
        """
        Author: Samuel Westlake and Alix Leroy
        Save the model weights to a file path given by the user
        :return: None
        """
        history = self.history.history
        epoch = self.history.epoch
        if file_path is None:
            metric = self.config["callbacks"]["checkpoint"]["args"]["monitor"]
            directory = "%s/%s" % (self.config["directory"], self.config["name"])
            try:
                metric = history[metric][history.index[epoch - 1]]
            except IndexError:
                metric = 0
            name = "%s_weights_%i_%.2f.h5" % (self.config["name"], epoch, metric)
            file_path = "%s/%s" % (directory, name)
        else:
            if to_working_dir:
                file_path = "%s/%s/%s" % (self.config["working_dir"], self.config["name"], file_path)
        if len(file_path.split("/")) > 1:                               # If the path contains a directory
            directory = "/".join(file_path.split("/")[:-1])             # Store the directory path
            os.makedirs(directory, exist_ok=True)                       # Make the directory if necessary
        self.net.save_weights(file_path)                              # Save weights
        print("Model saved to %s" % file_path)

    def evaluate(self, data_set=None):
        """
        Author: Samuel Westlake
        Uses DataGenerator to return batches of data for evaluation
        A CSV is saved with: image path, ground truth, prediction and a value to signify if the prediction was correct
        :return:
        """
        # Which data set to evaluate on and get the number of samples (val or surrender)
        if data_set is None:
            data_set = self.config["evaluate"]["data_set"]

        if self.config["data_gen"]["common"]["method"] == "from_directory":
            self.__evaluate_from_directory(data_set)
        elif self.config["data_gen"]["common"]["method"] == "from_file":
            self.__evaluate_from_file(data_set)

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
    def quit():
        quit()

    def __on_start(self):
        """
        Author: Samuel Westlake
        If on_start: enabled is True
        Runs commands given in on_start in the config with any given args and kwargs
        :return: None
        """
        if self.config["on_start"]["enabled"]:
            if self.config["on_start"]["commands"] is not None:
                for line in self.config["on_start"]["commands"]:
                    for cmd, params in line.items():
                        args, kwargs = self.__get_args_and_kwargs(params)
                        print("> %s(*%s, **%s)" % (cmd, args, kwargs))
                        getattr(self, cmd)(*args, **kwargs)

    def __config_update(self, new_values, path=[]):
        """
        Updates the config given a dict of new values to add/update
        :param new_values: dict
        :param path: [] for internal use
        :return: None
        """
        if isinstance(new_values, dict):
            for key, value in new_values.items():
                self.__config_update(value, path + [key])
        else:
            self.config = helpers.nested_set(self.config, path, new_values)


    def __evaluate_from_directory(self, data_set):
        args = {**self.config["data_gen"]["common"], **self.config["data_gen"][data_set]}
        data_gen = DataGenerator(**args)
        steps = data_gen.steps()
        epoch = self.history.epoch
        directory = "%s/%s" % (self.config["working_dir"], self.config["name"])
        predictions_file_name = "predictions_%s_%s.csv" % (epoch, data_set)
        incorrect_dir = "%s/%s/incorrect_%s_%s" % \
                        (self.config["working_dir"], self.config["name"], epoch, data_set)

        # Initialise data generator
        flow = data_gen.flow(yield_paths=True)

        # Evaluate and save incorrect images
        self.evaluation = {}
        with open("%s/%s" % (directory, predictions_file_name), "w") as file:
            file.write("path,ground truth,prediction,correct\n")
            for step in range(steps):
                x, y, image_paths = next(flow)
                y = np.argmax(y, axis=1)
                y_hat = self.net.predict_on_batch(x)
                y_hat = np.argmax(y_hat, axis=1)
                for image_path, ground_truth, prediction in zip(image_paths, y, y_hat):
                    correct = int(prediction == ground_truth)
                    file.write("%s,%i,%i,%i\n" % (image_path, ground_truth, prediction, correct))
                    self.__update_evaluation(ground_truth, prediction)
                    if not correct and self.config["evaluate"]["save_incorrect"]:
                        self.__copy_incorrect(image_path, incorrect_dir, ground_truth, prediction)
                helpers.progress_bar(step, steps, prefix="evaluating", suffix="complete", length=30)

        # Write to file
        print("\n")
        metrics_values = self.net.evaluate_generator(data_gen.flow(), steps=steps)
        for key, value in zip(self.net.metrics_names, metrics_values):
            self.evaluation[key] = value
        evaluation_file_name = "evaluation_%s_%s.txt" % (epoch, data_set)
        try:
            os.remove("%s/%s" % (directory, evaluation_file_name))
        except OSError:
            pass
        helpers.save_dict(self.evaluation, "%s/%s" % (directory, evaluation_file_name))
        helpers.print_dict(self.evaluation)

    def __evaluate_from_file(self, data_set):
        print("This method is missing")

    def __get_optimizer(self):
        """
        Author : Samuel Westlake
        Select the optimizer for the training
        :return: selected optimizer
        """
        opt_name = self.config["optimizer"]["algorithm"]
        params = None
        kwargs = self.config["optimizer"]["kwargs"]
        kwargs = kwargs if kwargs is not None else {}
        exec("import torch.optim as optim")
        exec("optimizer = optim.%s(%s, **%s)" % (opt_name, params, kwargs))

        return locals()["optimizer"]



    def __get_losses(self):
        """
        Author : Alix Leroy
        Get the list of standard and custom loss functions
        :return:
        """

        loss = []
        d = {}
        for i, l in enumerate(self.config["compile"]["loss"]):

            l = l.split(".")
            if len(l)== 2: # If we have "Module.custom_loss" we load the custom loss
                try:
                    exec("import sys")
                    exec('sys.path.append("..")')  # Adds higher directory to python modules path.
                    exec("from modules.loss.{0} import {1} as temp_loss".format( l[0], l[1]))
                    exec("d['loss{0}'] = temp_loss".format(helpers.convert_number_to_letters(i)))
                except:
                    raise ValueError("Cannot load the loss module")

                loss.append(d["loss{0}".format(helpers.convert_number_to_letters(i))])                                           # Add the custom loss to the list

            elif len(l)  > 2:
                raise ValueError("The custom loss specified cannot be found into the loss module")
            else:                                                                                                                # If it is a standard loss function, we directly load it
                loss.append(str(l[0]))

        return loss




    def __get_metrics(self):
        """
        Author : Alix Leroy
        Get the list of standard and custom metrics
        :return: list of metrics
        """

        metrics = []

        d = {}
        for i, m in enumerate(self.config["compile"]["metrics"]):

            m = m.split(".")
            if len(m) == 2:  # If we have "Module.metrics" we load the custom metric
                try:
                    exec("import sys")
                    exec('sys.path.append("..")')  # Adds higher directory to python modules path.
                    exec("from modules.metrics.{0} import {1} as temp_metric".format(m[0], m[1]))
                    exec("d['metric{0}'] = temp_metric".format(self.__convert_number_to_letters(i)))
                except:
                    raise ValueError("Cannot load the metric module")

                metrics.append(d["metric{0}".format(self.__convert_number_to_letters(i))])  # Add the custom metric to the list

            elif len(m) > 2:
                raise ValueError("The custom loss specified cannot be found into the loss module")
            else:  # If it is a standard loss function, we directly load it
                metrics.append(str(m[0]))

        return metrics




    def __initialize_checkpoint(self):
        log_dir = "%s/%s" % (self.config["working_dir"], self.config["name"])
        kwargs = self.config["callbacks"]["checkpoint"]["kwargs"]
        metric = kwargs["monitor"]
        if kwargs["save_weights_only"]:
            file_name = "%s_weights_{epoch:02d}_{%s:.2f}.h5" % (self.config["name"], metric)
        else:
            file_name = "%s_model_{epoch:02d}_{%s:.2f}.h5" % (self.config["name"], metric)
        return ModelCheckpoint(log_dir=log_dir,
                               file_name=file_name,
                               **kwargs)

    def __update_evaluation(self, ground_truth, prediction):
        """
        Updates the evaluation dictionary with a new ground truth and prediction
        :param ground_truth:
        :param prediction:
        :return:
        """
        correct = int(prediction == ground_truth)
        if ground_truth in self.evaluation:
            self.evaluation[ground_truth]["total"] += 1
            self.evaluation[ground_truth]["correct"] += correct
        else:
            self.evaluation[ground_truth] = {"total": 1, "correct": correct, "incorrect": {}}
        if not correct:
            if prediction in self.evaluation[ground_truth]["incorrect"]:
                self.evaluation[ground_truth]["incorrect"][prediction] += 1
            else:
                self.evaluation[ground_truth]["incorrect"][prediction] = 1
        for key in self.evaluation:
            self.evaluation[key]["accuracy"] = self.evaluation[key]["correct"] / self.evaluation[key]["total"]

    def __set_fit_epoch(self, fit_gen_kwargs):
        """
        This method takes care of the kwargs for the fit_generator.
        Kwargs initial_epoch, epoch require additional processing before training
        :return:
        """
        # Set initial epoch
        if fit_gen_kwargs["initial_epoch"] == "auto":
            fit_gen_kwargs["initial_epoch"] = self.history.epoch
        # Set epoch
        fit_gen_kwargs["epochs"] += fit_gen_kwargs["initial_epoch"]
        return fit_gen_kwargs

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


