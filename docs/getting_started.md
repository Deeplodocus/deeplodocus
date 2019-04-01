# Getting Started

## Before starting

Before starting, we recommend you to install both PyTorch and Deeplodocus. FOr more information please refer to the [installation page](deeplodocus.org/en/master/installation)


## Deeplodocus Admin

The first Deeplodocus tool to present is the Admin system. Once Deeplodocus is correctly installed, you will be able to use Deeplodocus directly from your terminal.

For example, if you want to start a new Deeplodocus project, you can do it by entering :

`deeplodocus startproject <my_project_name>`

In the above command `<my_project_name>` must be replaced by the name of your project.This command generates a full Deeplodocus project within the folder you are located.

For more information about the Admin commands, please refer to the [Admin Commands page](http:deeplodocus.org/en/master/admin_commands)

## Deeplodocus Structure

When using the command `deeplodocus startproject SkyNet` Deeplodocus will generate a project with a defined structure. In this section we will give you a quick overview of the generated project structure.

A "virgin" project comes with the following files and folders :

- `main.py`
- `config`
- `data`
- `modules`

### `main.py`

`main.py` is the main file to start your project. This file contains the following lines :

```python
#!/usr/bin/env python3

from deeplodocus.brain import Brain

brain = Brain(config_dir="./config")
brain.wake()
```

The goal of this file is to feed the Deeplodocus' Brain, which is the core of any Deeplodocus project, with a config folder. The brain is then awaken allowing you to start.
<span style="color:red"> ** /!\ We strongly recommend not to change anything in this file**</sapn>


### `config`

This folder contain all the configuration for your Deeplodocus project. It is this folder whose the relative path is given to the `main.py` as seen previously.

We will describe the content of this folder later. However, if you want to have a detailed description of the configuration files withing the `config` folder, please check our [Config page](deeplodocus.org/master/en/config)

### `data`

The data folder is an empty folder.
If dealing with small datasets, we recommend you to copy them into this folder.

### `modules`

In Deeplodocus a module can be a transformation operation, a deep network, a loss function, a metric, etc.
One particularity of Deeplodocus is the possibility to create and use your own custom modules. Once created, we recommend you to copy your module within the corresponding sub-folder into `modules`.


For more information on the existing sub-folder of `modules`, please check our [Modules page](deeplodocus.org/en/master/modules)


## Example # 1: MNIST

In this example, we focus on training a deep network to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

To do so, we need to create a Deeplodocus project :

`deeplodocus startproject MNIST`

In the `config/data.yaml` file, we can:
 
 - Select a batch of 32 images
 - Select 4 workers (processes) to load the images
 - Enable the training step
 - Enable the validation step 
 - Define 1 input located into the folder (the handwritten digit images)
 - Define 1 label (the label corresponding to the image)


The `config/data.yaml` looks as below :

```yaml
# config/data.yaml

dataloader:
  batch_size: 32              # 32 images per batch
  num_workers: 4              # 4 Processes
  
enabled:
  train: True                 # Enable The training step
  validation: True            # Enable the validation step
  
dataset:
  train:
    inputs:
      - source: "./data/..."    # Path to data file/folder
        join: Null              # Path to append to the start of each data item
        type: "image"           # One of: "integer", "image", "string", "float", "np-array", "bool", "video", "audio"
        load_method: "default"  # One of: "online" / "default" or "offline"

    labels:
      - source: "./data/..."    # Path to data file/folder
        join: Null              # Path to append to the start of each data item
        type: "int"             # One of: "integer", "image", "string", "float", "np-array", "bool", "video", "audio"
        load_method: "default"  # One of: "online" / "default" or "offline"
    name: "Training"
    
  validation:
    inputs:
      - source: "./data/test/images"    # Path to data file/folder
        join: Null                      # Path to append to the start of each data item
        type: "image"                   # One of: "integer", "image", "string", "float", "np-array", "bool", "video", "audio"
        load_method: "default"          # One of: "online" / "default" or "offline"

    labels:
      - source: "./data/test/labels.dat"    # Path to data file/folder
        join: Null                          # Path to append to the start of each data item
        type: "int"                         # One of: "integer", "image", "string", "float", "np-array", "bool", "video", "audio"
        load_method: "default"              # One of: "online" / "default" or "offline"
    name: "Validation"
```

Note: The unused key arguments are not displayed here

In the `config/transform.yaml` we can define :

- A transformer attached to the input image
- No transformer attached to the label

```yaml
train:
  name: "Train Transform Manager"
  inputs: "config/transforms/transform_input.yaml"
  labels: Null
validation:
  name: "Validation Transform Manager"
  inputs: "config/transforms/transform_input.yaml"
  labels: Null
```

The `tranform_input.yaml` file is defined as follow :

```yaml
# config/transforms/transform_input.yaml

method: "sequential"            # Type of transformer
name: "INput Image transformer" # Name of the transformer

mandatory_transforms_start: Null # List all the transforms to operate at start


transforms:                     # List all the transforms to operate
  - name: "normalize_image"     # Normalize the image
    module: Null                # The normalize_image module will be searched automatically by Deeplodocus
    kwargs:
      mean: 127.5
      standard_deviation: 255

mandatory_transforms_end: Null        # List all the transforms to operate at the end
```

Note : This transformer is a [Sequential Transformer](deeplodocus.org/en/master/transformer#sequential), The future version of deeplodocus will remove the mandatory transforms which are not necessary for this particular type of transformer.

We can then define the optimizer in `config/optimizer.yaml`:

```yaml
# config/optimizer.yaml

module: "torch.optim"   # Load a predefined optimizer from PyTorch
name: "Adam"            # Load Adam
kwargs: {}              # No arguments => Use the default parameters
```

```yaml

from_file: False
file: Null
module: "deeplodocus.app.models.lenet"
name: "LeNet"
input_size:
  - [1, 28, 28]
kwargs: {}
```



## Example # 2: CityScapes