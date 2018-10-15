from deeplodocus.data.transform_manager import TransformManager
from deeplodocus.utils.namespace import Namespace

# Get the config for the transform managers
config = Namespace(yaml_path="/home/alix/Documents/deeplodocus/deeplodocus/core/project/deep_structure/config/transform_config.yaml")


# config.summary()

config.train.summary()

# Create the transform managers
transform_manager_train = TransformManager(config.train, write_logs=False)
transform_manager_val = TransformManager(config.validation, write_logs=False)
transform_manager_test = TransformManager(config.test, write_logs=False)