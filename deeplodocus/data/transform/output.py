from deeplodocus.utils.namespace import Namespace
from deeplodocus.utils.generic_utils import get_module
from deeplodocus.flags import DEEP_MODULE_TRANSFORMS
from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import DEEP_NOTIF_FATAL


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
            for transform_name, transform_info in sequence.transforms.get().items():
                method, module = get_module(
                    **transform_info.get(ignore="kwargs"),
                    browse=DEEP_MODULE_TRANSFORMS
                )
                transform_info.add({"method": method})
                transform_info.module = module

    def transform(self, outputs):
        """
        Calls the appropriate transform method dependant on the number of transformers and outouts
        1 output -> __transform_single_output
        2+ outputs and 1 transformer -> __transform_multi_output_series
        2+ outputs and 2+ transformers -> __transform_multi_output_mapped
        :param outputs: torch.tensor or [torch.tensor]: outputs from the model
        :return: torch.tensor or [torch.tensor]: transformed outputs
        """
        # Multiple model outputs
        if isinstance(outputs, list) and len(outputs) > 1:
            # 0 or 1 transformers - apply transformer to each output
            if len(self.output_transformer) < 2:
                return self.__transform_multi_output_series(outputs)
            # If more than 1 output transformer, there must be numbers of transformers and outputs
            # Map by order
            elif len(self.out_transformer) == len(outputs):
                return self.__transform_multi_output_mapped(outputs)
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
            return self.__transform_single_output(outputs)

    def __transform_multi_output_series(self, outputs: list) -> list:
        """
        Apply each transform in each transformer to each output
        :param outputs: [torch.tensor]: output from the model
        :return: [torch.tensor]: transformed outputs
        """
        for i, output in enumerate(outputs):
            # Apply each output transform from each transformer sequence to the output
            for sequence in self.output_transformer:
                for _, transform in sequence.transforms.get().items():
                    output = transform.method(output, **transform.kwargs.get())
            # Update the output
            outputs[i] = output
        return outputs

    def __transform_multi_output_mapped(self, outputs: list) -> list:
        """
        Apply each transformer to the corresponding output tensor
        :param outputs: [torch.tensor]: outputs from the model
        :return: [torch.tensor]: transformed outputs
        """
        # For each output and transformer
        for i, (output, transformer) in enumerate(zip(outputs, self.transformer)):
            # Apply each transform to the output
            for _, transform in transformer.transforms.get().items():
                output = transform.method(output, **transform.kwargs.get())
            # Update the output
            outputs[i] = output
        return outputs

    def __transform_single_output(self, output):
        """
        Apply the transformer or series of transformers to the output tensor
        :param output: (torch.tensor) or ([torch.tensor] with len == 1): outputs from the model
        :return: (torch.tensor) or ([torch.tensor] with len == 1): transformed outputs
        """
        for sequence in self.output_transformer:
            for _, transform in sequence.transforms.get().items():
                output = transform.method(output, **transform.kwargs.get())
        return output

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
