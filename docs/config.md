# About Deeplodocus Configurations

The configurations detailed by files in the config directory are the core of any Deeplodocus project - these manage each of the different modules and how they interact. They are split into nine different files, each responsible for a different domain of the project, as follows:

- [Project](config.md#project) - configure overarching project variables
- [Data](config.md#data) - configure your datasets
- [Model](config.md#model) - set up your model
- [Losses](config.md#losses) -configure your loss functions.
- [Metrics](config.md#metrics) - pconfigure any metric functions
- [Transform](config.md#transform) - configure your transformation routines
- [Optimizer](config.md#optimizer) - configure your optimizer
- [Training](config.md#training) - set your training conditions
- [History](config.md#history) - configurations for storing training and validation history

On startup, each of the configuration files are loaded into a single variable named config and the datatype of each entry is checked. If any entries are missing or cannot be converted to the expected data type, they will be added/corrected with a default value and a warning will be issued to explain the change made.

Config is a Namespace object, more details of which can be found in the [Namespace section of the API page](config.md#namespace). Users have direct acces to the config variable via Deeplodocus terminal, for example:

- The [summary](api.md#summary) method can be used to view config - `config.summary()`
- A [snapshot](api.md#snapshot) of the config file can be taken - `config.snapshot()`
- Sub-domains can be viewed individually - `config.project.summary()`
- Variables can be edited - `config.training.num_epochs = 10`

The following sections detail all of the expected entries for each configuration YAML file. 

## Project

Overarching project variables are specified in the project.yaml file.

```yaml
# Example project.yaml file
session: "version01"
cv_library: "opencv"
device: "auto"
device_ids: "auto"
logs:
  history_train_batches: True
  history_train_epochs: True
  history_validation: True
  notification: True
on_wake: Null
```

#### session

The name of the current Deeplodocus session. History, log and weight files will be saved to a directory named with the session variable. 

- **Data type:** str
- **Default value:** "version01" 

#### cv_library
	
The computer vision library to use. 

- **Data type:** str
- **Default value:** "opencv" 
- **Supported options:**
	- For OpenCV use: "opencv"
	- For PILLOW use: "pil" 
	
#### device

The hardware device to use for executing forward and backward passes of the model.

- **Data type:** str
- **Default value:** "auto" 
- **Supported options:**
	- For CPU use: "cpu"
	- For CUDA devices use: "cuda"
	- "auto" will use CUDA devices if any are available, otherwise CPUs

#### device_ids

 The index values of CUDA devices to use.

- **Data type:** [int]
- **Default value:** "auto"
- **Supported options:**
	-  "auto" will use all avaialble CUDA devices
	- [0, 1, ... n] will use CUDA devices which have index values in the given list
	
#### logs: history_train_batches

Whether or not training loss and metric values for each batch should be written to a history CSV file.

- **Data type:** bool
- **Default value:** True

#### logs: history_train_epochs

Whether or not training loss and metric values for each epoch should be written to a history CSV file.

- **Data type:** bool
- **Default value:** True

#### logs: history_validation

Whether or not validation loss and metric values for each epoch should be written to a history CSV file.

- **Data type:** bool
- **Default value:** True

#### logs: notification

Whether or not all Deeplodocus notifications.

- **Data type:** bool
- **Default value:** True

#### on_wake

A list of commands to run on startup.

- **Data type:** [str]
- **Default value:** None

## Data

```yaml
# Example data.yaml file
TODO
```

## Model

A single model can be specified in the model.yaml file.

```yaml
# Example model.yaml file
name: "VGG16"
module: "deeplodocus.app.models.vgg"
from_file: False
file: Null
input_size: 
  - [3, 244, 244]
kwargs: Null
```

#### name

The name of the model to use. 

- **Data type:** str
- **Default value:** VGG16

#### module

Path to the module that contains the chosen model. If no module is given, Deeplodocs will search through deeplodocus.app and existing PyTorch modules for classes and functions with names that match the one given. The first object found with the requested name will be loaded and the user will be informed of its origin through a notification.

- **Data type:** str
- **Default value:** None

<<<<<<< HEAD
Note: More informaton about pre-built deeplodocus models can be found [here](existing_modules.md#models).
=======
Note: More informaton about pre-built deeplodocus models can be found [here](existing_modules/models.md).
>>>>>>> hotfix-0.3.1

#### from_file

If 'from_file' is true the model will be loaded from existing a pre-trained model/weight file specified by 'file'. Otherwise, the model will be loaded from the given 'name' and 'module' only.

- **Data type:** bool
- **Default value:** False

#### file

If 'from_file' is True, the model will be loaded according to the path to a model/weights file specified by 'file'. If a weights file contains the module path and name of its model, Deeplodocus will automatically attempt to use that model, otherwise the name and module specified in model.yaml will be used as the model.

- **Data type:** str
- **Default value:** None

#### input_size

The shape of each input to the model. Models may have multiple inputs, thus this entry is list of lists. 'input_size' is not a obligatory entry, but is required to print summaries of the model.

- **Data type:** [[int]]
- **Default value:** None

#### kwargs

Any keyword arguments to be parsed to the model.

- **Data type:** dict
- **Default value:** None

## Losses

Any number of losses can be specified in losses.yaml. Give each loss a unique name, followed by a series of defining entries, as seen below. The unique name given will be used when displaying loss values and saving to training and validation history.

```yaml
# Example loss.yaml file
LossName:
  name: "CrossEntropyLoss"
  module: Null
  weight: 1
  kwargs: Null
  
AnotherLossName:
  name: "CrossEntropyLoss"
  module: Null
  weight: 1
  kwargs: Null
```

#### name

The name of the loss object to use. 

- **Data type:** str
- **Default value:** "CrossEntropyLoss"

#### module

Path to the module that contains the chosen loss. If no module is given, Deeplodocs will search through deeplodocus.app and existing PyTorch modules for loss functions with names that match the one given. The first object found with the requested name will be loaded and the user will be informed of its origin through a notification.

- **Data type:** str
- **Default value:** None

<<<<<<< HEAD
Note: More informaton about existing PyTorch loss functions can be found [here](https://pytorch.org/docs/stable/nn.html#loss-functions), and some additional deeplodocus losses can be found [here](existing_modules.md#losses).
=======
Note: More informaton about existing PyTorch loss functions can be found [here](https://pytorch.org/docs/stable/nn.html#loss-functions), and some additional deeplodocus losses can be found [here](existing_modules/losses.md).
>>>>>>> hotfix-0.3.1

#### weight

The loss weight.

- **Data type:** float
- **Default value:** 1

#### kwargs

Any keyword arguments to be parsed to the loss.

- **Data type:** dict
- **Default value:** None

## Metrics

Any number of metrics can be specified in metrics.yaml. Give each metric a unique name, followed by a series of defining entries, as seen below. The unique name given will be used when displaying metric values and saving to training and validation history.

```yaml
# Example metrics.yaml file
MetricName:
  name: "accuracy"
  module: Null
  kwargs: Null
  
AnotherMetricName:
  name: "accuracy"
  module: Null
  kwargs: Null
```

#### name

The name of the metric object to use. 

- **Data type:** str
- **Default value:** "accuracy"

#### module

Path to the module that contains the chosen loss. If no module is given, Deeplodocs will search through deeplodocus.app and existing PyTorch modules for loss functions with names that match the one given. The first object found with the requested name will be loaded and the user will be informed of its origin through a notification.

- **Data type:** str
- **Default value:** None

<<<<<<< HEAD
Note: More informaton about existing metrics that come with Deeplodocus can be found [here](existing_modules.md#metrics).
=======
Note: More informaton about existing metrics that come with Deeplodocus can be found [here](existing_modules/metrics.md).
>>>>>>> hotfix-0.3.1

#### kwargs

Any keyword arguments to be parsed to the metric.

- **Data type:** dict
- **Default value:** None

## Transform

<<<<<<< HEAD
A series of input, label, additional data and output transformers can be specified in the transform.yaml file. 
=======
A series of input, label, additional data and output transformers can be specified in the transform.yaml file. More informaton about existing transforms can be found [here](existing_modules/transforms.md).
>>>>>>> hotfix-0.3.1

```yaml
# Example transform.yaml file
train:
  name: Train Transform Manager
  inputs: Null
  labels: Null
  additional_data: Null
  outputs: Null
validation:
  name: Validation Transform Manager
  inputs: Null
  labels: Null
  additional_data: Null
  outputs: Null
test:
  name: Test Transform Manager
  inputs: Null
  labels: Null
  additional_data: Null
  outputs: Null
predict:
  name: Prediction Transform Manager
  inputs: Null
  labels: Null
  additional_data: Null
  outputs: Null
```

#### train: name

A name transform manager dedicated to the training pipeline.

- **Data type:** str
- **Default value:** Train Transform Manager

#### validation: name

A name transform manager dedicated to the validation pipeline.

- **Data type:** str
- **Default value:** Validation Transform Manager

#### test: name

A name transform manager dedicated to the testing pipeline.

- **Data type:** str
- **Default value:** Test Transform Manager

#### predict: name

A name transform manager dedicated to the prediction pipeline.

- **Data type:** str
- **Default value:** Prediction Transform Manager

#### inputs

For each of the train, test, validation and predict pipelines, you can specify a path to a transformer YAML file for each of the model inputs. 

- **Data type:** [str]
- **Default value:** Null

#### labels

- **Data type:** [str]
- **Default value:** Null

For each of the train, test, validation and predict pipelines, you can specify a path to a transformer YAML file for each of the model labels. 

#### additional_data

- **Data type:** [str]
- **Default value:** Null

For each of the train, test, validation and predict pipelines, you can specify a path to a transformer YAML file for each of the model additional.

#### outputs

For each of the train, test, validation and predict pipelines, you can specify a path to a transformer YAML file for each of the model output. 

- **Data type:** [str]
- **Default value:** Null

## Optimizer

A single optimizer for the model should be specified in optimizer.yaml.

```yaml
# Example optimizer.yaml file
name: Adam
module: Null
kwargs: Null
```

#### name

The name of the optimizer to use. 

- **Data type:** str
- **Default value:** "Adam"

#### module

Path to the module that contains the chosen loss. If no module is given, Deeplodocs will search through deeplodocus.app and existing PyTorch modules for loss functions with names that match the one given. The first object found with the requested name will be loaded and the user will be informed of its origin through a notification.

- **Data type:** str
- **Default value:** None

Note: More informaton about existing PyTorch optimizers can be found [here](https://pytorch.org/docs/stable/optim.html#algorithms).

#### kwargs

Any keyword arguments to be parsed to the optimizer.

- **Data type:** dict
- **Default value:** None

## Training

```yaml
# Example training.yaml file
num_epochs: 10
initial_epoch: 0
shuffle: "default"
saver:
  method: "pytorch"
  save_signal: "auto"
  overwrite: False
overwatch:
  name: "Total Loss"
  condition: "less"
```
#### num_epochs

The epoch number to train to.

- **Data type:** int
- **Default value:** 10

#### initial_epoch

The number of the initial epoch.

- **Data type:** int
- **Default value:** 0

#### shuffle

How the training dataset should be shuffled.

- **Data type:** str
- **Default value:** "default"
- **Supported options:** 
	- "none" - no shuffling
	- "all" / "default" - all instances are shuffled at the start of each epoch.
	- "batches" - instances remain in the same batch, and the order of the batches is shuffled.
	- "pick" - instances are randomly selected from the dataset.
	
Note: Use "pick" when wanting to shuffle a dataset whilst simultaneously restricting the size of the dataset through the number entry in the data.yaml file.

#### saver: method

The format to save the the model as.

- **Data type:** str
- **Default value:** "pytorch"
- **Supported options:** 
	- "pytorch" - currenty the only supported option is saving as a pytorch weights file.

#### saver: save_signal

How regularly a signal to save the model weights should be dispatched. 

- **Data type:** str
- **Default value:** "auto"
- **Supported options:** 
	- "auto" - the model will be saved at the end of each epoch if the overwatch condition is met.
	- "batch" - the model will be saved at the end of each batch.
	- "epoch" - the model will be saved at the end of each epoch.

#### saver: overwrite

Whether or not the model weights should be overwritten on each save signal.

- **Data type:** bool
- **Default value:** False

#### overwatch: name

The name of the overwatch metric to watch. This can be the name of any loss or metric in use, or set to "Total Loss" to watch the weighted sum of all losses.

- **Data type:** str
- **Default value:** "Total Loss"

#### overwatch: condition

The condition for comparing previous overwatch metric values with the current value.

- **Data type:** str
- **Default value:** "less"
- **Supported options:** 
	- "<", "smaller", "less" - The current overwatch metric must be lower than the previous lowest.
	- ">", "bigger", "greater", "more" - The current overwatch metric must be greater than the previous greatest.

## History

```yaml
# Example history.yaml file
verbose: "default"
memorize: "batch"
```

#### verbose

TODO: description

- **Data type:** str
- **Default value:** "default"

#### memorize

TODO: description

- **Data type:** str
- **Default value:** "batch"

