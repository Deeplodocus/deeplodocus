from deeplodocus.data.transform_manager import TransformManager
from deeplodocus.utils.namespace import Namespace

# Get the config for the transform managers
config_transforms = Namespace(yaml_path="/home/alix/Documents/deeplodocus/deeplodocus/core/project/deep_structure/config/transform_config.yaml")


# Create the transform managers
transform_manager_train = TransformManager(config_transforms.train, write_logs=False)
#transform_manager_val = TransformManager(config_transforms.validation, write_logs=False)
#transform_manager_test = TransformManager(config_transforms.test, write_logs=False)


