import inspect
import types

from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.flags import DEEP_MODULE_TRANSFORMS
from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import *


class OutputTransformer(Namespace):

    def __init__(self, transform_files=None):
        super(OutputTransformer, self).__init__({"output_transformer": []})
        if transform_files is not None:
            for file in transform_files:
                self.output_transformer.append(Namespace(file))
            self.load_transform_functions()

    def load_transform_functions(self):
        """
        Calls the get_module function for each transform in each transformer
        Loads the returned method and edits the module entry to reflect the origin
        :return: None
        """
        # For each transform sequence
        for i, sequence in enumerate(self.output_transformer):
            # Check that the transforms entry exists
            self.__check_transforms_exists(sequence, i)
            if sequence.transforms is not None:
                for transform_name, transform_info in sequence.transforms.get().items():
                    self.__transform_loading(transform_name, transform_info)
                    method, module = get_module(
                        **transform_info.get(ignore="kwargs"),
                        browse=DEEP_MODULE_TRANSFORMS
                    )
                    if method is None:
                        self.__method_not_found(transform_info)
                    if isinstance(method, types.FunctionType):
                        transform_info.add({"method": method})
                    else:
                        try:
                            transform_info.add(
                                        {"method": method(**transform_info.kwargs.get())}
                                    )
                        except TypeError as e:
                            Notification(DEEP_NOTIF_FATAL, str(e))
                    transform_info.module = module
                    self.__transform_loaded(transform_name, transform_info)

    def transform(self, outputs, inputs=None, labels=None, additional_data=None):
        """
        Calls the appropriate transform method dependant on the number of transformers and outputs
        1 output -> __transform_single_output
        2+ outputs and 1 transformer -> __transform_multi_output_series
        2+ outputs and 2+ transformers -> __transform_multi_output_mapped
        :param outputs: torch.tensor or [torch.tensor]: outputs from the model
        :param inputs:
        :param labels:
        :param additional_data:
        :return: torch.tensor or [torch.tensor]: transformed outputs
        """
        # Multiple model outputs
        if isinstance(outputs, list) and len(outputs) > 1:
            # 0 or 1 transformers - apply transformer to each output
            if len(self.output_transformer) < 2:
                return self.__transform_multi_output_series(
                    outputs,
                    inputs=inputs,
                    labels=labels,
                    additional_data=additional_data
                )
            # If more than 1 output transformer, there must be numbers of transformers and outputs
            # Map by order
            elif len(self.out_transformer) == len(outputs):
                return self.__transform_multi_output_mapped(
                    outputs,
                    inputs=inputs,
                    labels=labels,
                    additional_data=additional_data
                )
            # Else, cannot map outputs to transformers
            else:
                Notification(
                    DEEP_NOTIF_FATAL,
                    "Cannot map %i model outputs to %i transformers" % (
                        len(self.output_transformer), len(outputs)
                    )
                )
        # Single model output
        else:
            return self.__transform_single_output(
                outputs,
                inputs=inputs,
                labels=labels,
                additional_data=additional_data
            )

    def finish(self):
        for sequence in self.output_transformer:
            for _, transform in sequence.transforms.get().items():
                if hasattr(transform.method, "finish"):
                    transform.method.finish()

    def __transform_multi_output_series(self,
                                        outputs: list,
                                        inputs=None,
                                        labels=None,
                                        additional_data=None
                                        ) -> list:
        """
        Apply each transform in each transformer to each output
        :param outputs: [torch.tensor]: output from the model
        :return: [torch.tensor]: transformed outputs
        """
        for i, output in enumerate(outputs):
            # Apply each output transform from each transformer sequence to the output
            for sequence in self.output_transformer:
                for _, transform in sequence.transforms.get().items():
                    output = self.__apply(
                        transform,
                        output,
                        inputs=inputs,
                        labels=labels,
                        additional_data=additional_data
                    )
            # Update the output
            outputs[i] = output
        return outputs

    def __transform_multi_output_mapped(self,
                                        outputs: list,
                                        inputs=None,
                                        labels=None,
                                        additional_data=None,
                                        ) -> list:
        """
        Apply each transformer to the corresponding output tensor
        :param outputs: [torch.tensor]: outputs from the model
        :return: [torch.tensor]: transformed outputs
        """
        # For each output and transformer
        for i, (output, transformer) in enumerate(zip(outputs, self.transformer)):
            # Apply each transform to the output
            for _, transform in transformer.transforms.get().items():
                output = self.__apply(
                    transform,
                    output,
                    inputs=inputs,
                    labels=labels,
                    additional_data=additional_data
                )
            # Update the output
            outputs[i] = output
        return outputs

    def __transform_single_output(self,
                                  outputs,
                                  inputs=None,
                                  labels=None,
                                  additional_data=None
                                  ):
        """
        Apply the transformer or series of transformers to the output tensor
        :param outputs: (torch.tensor) or ([torch.tensor] with len == 1): outputs from the model
        :return: (torch.tensor) or ([torch.tensor] with len == 1): transformed outputs
        """
        for sequence in self.output_transformer:
            for _, transform in sequence.transforms.get().items():
                output = self.__apply(
                    transform,
                    outputs,
                    inputs=inputs,
                    labels=labels,
                    additional_data=additional_data
                )
        return output

    @staticmethod
    def __apply(transform, outputs, **kwargs):
        """
        Applies transform method to output and returns the transformed output
        If method is a class
            Ues method.forward(output)
        Else
            Uese method(output, kwargs)
        :param transform: Namespace: transform to be applied
        :param output: output to be transformed
        :return: transformed output
        """
        # If transform is a class (has a forward attribute)
        if hasattr(transform.method, "forward"):
            # Remove non-required and None entries from kwargs
            kwargs = {
                key: item for key, item in kwargs.items()
                if item is not None and key in inspect.getargspec(transform.method.forward)[0]
            }
            outputs = transform.method.forward(outputs=outputs, **kwargs)
        # If transform is a function
        else:
            kwargs = {
                key: item for key, item in kwargs.items()
                if item is not None and key in inspect.getargspec(transform.method)[0]
            }
            outputs = transform.method(
                outputs=outputs,
                **{**kwargs, **transform.kwargs.get()}
            )
        return outputs

    @staticmethod
    def __check_transforms_exists(sequence, i=None):
        """
        Calls a DEEP_NOTIF_FATAL if 'transforms' does not exist in the given sequence
        :param sequence: str: name of the sequence (to help the user debug)
        :param i: int: index of the sequence (to help user debug)
        :return: None
        """
        if "transforms" not in sequence.get():
            msg = "'transforms' entry not found in '%s'" % sequence.name
            if i is not None:
                msg += ", (output_transformer[%i])" % i
            Notification(DEEP_NOTIF_FATAL, msg)

    @staticmethod
    def __method_not_found(info):
        msg = "Transform method not found : %s" % info.name
        if info.module is not None:
            msg += " from %s" % info.module
        Notification(DEEP_NOTIF_FATAL, msg)

    @staticmethod
    def __transform_loading(name, info):
        """
        Notify user about the transform about to be loaded
        :param name: str: Name given to the transform by the user
        :param info: Namespace: information about the transform
        :return: None
        """
        msg = "Loading transform : %s : %s" % (name, info.name) if info.module is None \
            else "Loading transform : %s : %s from %s" % (name, info.name, info.module)
        Notification(DEEP_NOTIF_INFO, msg)

    @staticmethod
    def __transform_loaded(name, info):
        """
        Notify the user that a transform has been loaded successfully
        :param name: str: Name given to the transform by the user
        :param info: Namespace: information about the transform
        :return: None
        """
        msg = "Loaded transform : %s : %s from %s" % (name, info.name, info.module)
        Notification(DEEP_NOTIF_SUCCESS, msg)
