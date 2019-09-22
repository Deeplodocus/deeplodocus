# MNIST with LeNet

#### Introduction
[MNIST](http://yann.lecun.com/exdb/mnist/) is a dataset of handwritten digits images and their corresponding labels. The training set contains 60.000 examples and the test set contains 10.000 examples. Image classification on the MNIST dataset is often considered as the "Hello World!" of Machine Learning. Yet, it does not mean the task is not complex at all.

In this tutorial we will explain how to perform the following :
- Start a Deeplodocus project
- Define the dataset
- Define some transform operation on the data using a `Some Of Transformer`
- Define the Model (based on LeNet)
- Define the Loss function
- Define a Metric
- Define the Optimizer
- Define the project (logs, cv_library, notifications, devices to perform inference)

#### Start a Deeplodocus project

Before starting a Deeplodocus project, please make sure you already installed the [PyTorch](pytorch.org) version corresponding to your installation.
Then, you can install Deeplodocus using the following command :

```bash
pip install deeplodocus
```
Once Deeplodocus is install, navigate to the directory you would like to create the project and enter :

```bash
deeplodocus start-project MNIST
```

The Deeplodocus project is now created ! :)

#### Define the Data
First we want to load the images. To do so, we can do it using two major main :
 
 - Use the PyTorch Dataset already existing in the `torchvision` package
 - Download the dataset ourselves, extract it and load it using Deeplodocus Source classes

For this tutorial, we will focus on the simpliest solution : the `torchvision` package.

Everything related to data loading is located into the `data.yaml` file. 
The first parameters we can add are :

- the minibatch size : number of instances processed at once by the model
- the number of workers : When taking advantage of multiprocessing we can choose how many CPU cores are used to load data
- enable tasks : We can choose to enable the training task and also the validation. Test and Predict will be put away for another tutorial
```yaml
# data.yaml
dataloader:
  batch_size: 2 # Batch of 2 images
  num_workers: 1  # 1 Worker loading the images
  enabled:
    train: True         # We train the network
    validation: True    # We validate the network during the training
    test: False
    predict: False
```
Still in the `data.yaml` file, we can now define what data to load. This is a bit verbose but don't be afraid, the following approach has two major advantages:
- It allows the user to understand and define every single parameter
- It is exactly the same structure for any Deeplodocus project, even complex ones

Just bellow `enable` in `data.yaml`, you will be able to add `datasets`. We will define 2 `datasets`: one for the training and one for the validation.
The two dataset will contain 2 `entries` : one for the image and one for the label.
Each entry will contain one source. The first entry contains a MNIST source loading both the image and its label, yet only giving the image to the network. The second entry contains a SourcePointer accessing the label from the loaded data in the first entry.

To load the first entry we have :
```yaml
# data.yaml
datasets:

#
# TRAINING SET
#
- name: "Train MNIST"                     # Name of the dataset
  type: "train"                           # Type of Dataset (train, validation, test, predict)
  num_instances: 60000                     # Num of instances (60.000 in the training set)
    entries:
        -   name: "MNIST image"           # We define the first entry
            type: "input"                 # It is an input
            data_type: "image"            # We load this data as an image
            load_as: "float16"            # And convert it to float16 before feeding the network (required for CNN) 
            move_axis: Null               # WE do not change the axes
            enable_cache: true            # Enable the cache to let the SourcePointer access the label
            sources :                     # Load the PyTorch dataset
                - name: "MNIST"
                  module: "torchvision.datasets"
                  kwargs:
                    root: "./MNIST"
                    train: true
                    download: true
```

To load the second entry we have :
```yaml
# data.yaml
-   name: "MNIST label"
    type: "label"             # This is a label => Only given to the loss function and the metrics
    data_type: "integer"      # Load it as an integer
    load_as: "int8"           # COnvert it to a int8
    move_axis: Null
    enable_cache: false       # No need to enable the cache here
    sources :                 # Create a SourcePointer to load the second item from the first source of the first entry
        - name: "SourcePointer"
          module: Null
          kwargs:
            entry_id: 0
            source_id: 0
            instance_id: 1
```

Similarly we can define the validation set :

```yaml
# data.yaml

#
# VALIDATION SET
#
- name: "Validate MNIST"
  type: "validation"
  num_instances: Null
    entries:
        -   name: "MNIST image"
            type: "input"
            data_type: "image"
            load_as: "float16"
            move_axis: Null
            enable_cache: true
            sources :
                - name: "MNIST"
                  module: "torchvision.datasets"
                  kwargs:
                    root: "./MNIST"
                    train: false
                    download: true

        -   name: "MNIST label"
            type: "label"
            data_type: "integer"
            load_as: "int8"
            move_axis: Null
            enable_cache: false
            sources :
                - name: "SourcePointer"
                  module: Null
                  kwargs:
                    entry_id: 0
                    source_id: 0
                    instance_id: 1
```


To summarize the `data.yaml` file, we have the following :
```yaml
#data.yaml 
dataloader:
  batch_size: 2 # Batch of 2 images
  num_workers: 1  # 1 Worker loading the images
  enabled:
    train: True         # We train the network
    validation: True    # We validate the network during the training
    test: False
    predict: False
  datasets:
  
    #
    # TRAINING SET
    #
    - name: "Train MNIST"
      type: "train"
      num_instances: 60000
        entries:
            -   name: "MNIST image"
                type: "input"
                load_as: "image"
                convert_to: "float32"
                move_axis: [2, 1, 0]
                enable_cache: true
                sources :
                    - name: "MNIST"
                      module: "torchvision.datasets"
                      kwargs:
                        root: "./MNIST"
                        train: true
                        download: true
    
            -   name: "MNIST label"
                type: "label"
                load_as: "integer"
                convert_to: "int64"
                move_axis: Null
                enable_cache: false
                sources :
                    - name: "SourcePointer"
                      module: Null
                      kwargs:
                        entry_id: 0
                        source_id: 0
                        instance_id: 1
    
    #
    # VALIDATION SET
    #
    - name: "Validate MNIST"
      type: "validation"
      num_instances: Null
        entries:
            -   name: "MNIST image"
                type: "input"
                load_as: "image"
                convert_to: "float32"
                move_axis: [2, 1, 0]
                enable_cache: true
                sources :
                    - name: "MNIST"
                      module: "torchvision.datasets"
                      kwargs:
                        root: "./MNIST"
                        train: false
                        download: true
    
            -   name: "MNIST label"
                type: "label"
                load_as: "integer"
                convert_to: "int64"
                move_axis: Null
                enable_cache: false
                sources :
                    - name: "SourcePointer"
                      module: Null
                      kwargs:
                        entry_id: 0
                        source_id: 0
                        instance_id: 1
```


You made it ! Let now define transform operations.

#### Define the Data Transformations

We want to perform data transformations during the traning. To do so we can define transformers in the `transform.yaml`. 
To do so, we define a `TransformManager` for the training and one for the validation. Here the two `TransformManager` contain a `Transformer` for the input data.


```yaml
#transforms.yaml
train:
  name: "Train Transform Manager"
  inputs: [./config/transforms/input.yaml]
  labels: Null
  additional_data: Null
  outputs: Null
validation:
  name: "Validation Transform Manager"
  inputs: [./config/transforms/input.yaml]
  labels: Null
  additional_data: Null
  outputs: Null
```

We can now define the `input.yaml`.

In `input.yaml` we want to normalize the input images. To do so we use a `Sequential` transformer.

```yaml
#./transforms/input.yaml
method: "sequential"
name: "Sequential transformer for MNIST input"

transforms:
  - normalize:
    name: normalize_image
    module:
    kwargs:
      standard_deviation: 255
      mean: 127.5


```

#### Define the Model
Then in `model.yaml`, we can load the model based on LeNet

```yaml
#model.yaml
from_file: False
file: Null
module: "deeplodocus.app.models.lenet"
name: "LeNet"
input_size:
  - [1, 28, 28]
kwargs: Null
```

#### Define the Loss function

We choose the `CrossEntropyLoss` as loss function.

```yaml
# losses.yaml
CrossEntropyLoss:
  module: Null
  name: "CrossEntropyLoss"
  weight: 1
  kwargs: {}
```

#### Define the Metrics
We also want to monitor the evolution of the accuracy during the training.

```yaml
#metrics.yaml
accuracy:
  module: Null
  name: "accuracy"
  weight: 1
  kwargs: {}
```
#### Define the Optimizer

We can choose Adam as an optimizer.

```yaml
#optimizer.yaml
module: "torch.optim"
name: "Adam"
kwargs:
  lr: 0.001
  eps: 1e-09
  amsgrad: False
  betas:
    - 0.9
    - 0.999
  weight_decay: 0.0
```

#### Define the project

Finally we can enable the logs and let Deeplodocus define the devices for inference.

```yaml
session: "version01"
cv_library: "opencv"
device: "auto"
device_ids: "auto"
logs:
  history_train_batches: True
  history_train_epochs: True
  history_validation: True
  notification: True
on_wake: []
```