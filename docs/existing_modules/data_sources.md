# Data Sources

Deeplodocus allows you to load data from different sources. Within the Deeplodocus platform, the sources are refered as `Source` subclasses.

Because Deeplodocus uses PyTorch as main backend, it takes advantage of the PyTorch multiprocessing `Dataloader` class. Therefore, Deeplodocus's sources are based on the PyTorch `Dataset` format.

Before reading this section, we advise you to have a look at the Dataset architecture of Deeplodocus explained in the [GETTING STARTED](getting_started.md#Data_Architecture) section.
## Use a Deeplodocus Source

#### File

To load data from a file (images, videos, integer, floats, numpy array, etc...) please use the following:
```yaml
sources:
    - module: "File"
      origin: Null
      kwargs:
        path: "filepath.txt"
```
If you want to load a sequence of items (e.g. a list of integer). You can use the `delimiter` argument:
```yaml
sources:
    - module: "File"
      origin: Null
      kwargs:
        path: "filepath.txt"
        delimiter: "," # Useful if we load sequences of items
```

This will work with any sequence of items (e.g. with integers: (1, 1, 1, 1) will be loaded as a list of 4 integers)



#### Folder

To load files (text, images, videos, etc...) from a directory (and its subdirectories), please use the following:

```yaml
sources:
    - module: "Folder"
      origin: Null
      kwargs:
        path: "/home/Deeplodocus/Pictures"
```
NOTE: In order to load data from a directory, Deeplodocus will optimize the data loading by generating a file listing all the items within the directory (and its subdirectories) and reading this file at every iteration. 


#### Camera
TODO

## Use a PyTorch Source

#### Directly use a PyTorch source
You can access any PyTorch source instance. 

For example, to access `MNIST` from `torchvision.datasets` simply use the following :

```yaml
sources :
    - module: "MNIST"
      origin: "torchvision.datasets"
      kwargs:
        root: "./MNIST"
        train: true
        download: true

```
A Source will only load one item (or sequence of items) of a specific type.
However, MNIST and many other PyTorch sources return multiple data. In the case of MNIST, calling the __getitem__() function returns a tuple (image, label). 
By default Deeplodocus will  load the first item (the image for MNIST) and will ignore the other items. To avoid such an issue, please use the `SourcePointer` as described below.
 

#### Use the SourcePointer

By default, Deeplodocus only takes the first item when loading data from a PyTorch dataset. The `SourcePointer` helps accessing the additional items without loading the data twice.

The `SourcePointer` points to a specific item loaded by a specific `Source` within a specific `Entry`. 
In order to access the label from the tuple (image, label) loaded from MNIST, you can use the following.

```yaml
sources :
    - module: "SourcePointer"
      origin: Null
      kwargs:
        entry_id: 0     # Access the first entry
        source_id: 0    # Access the first source of the first entry
        instance_id: 1  # Access the second item (label)
```
NOTE: Accessing data using a `SourcePointer` instance requires to enable the cache memory in the targeted `Entry`. 
To access the label from MNIST, please set `enable_cache: true` where the data is initially loaded:

```yaml
    entries:
        -   name: "MNIST image"
            type: "input"
            load_as: "image"
            convert_to: "float16"
            move_axis: Null
            enable_cache: true            # Set enable_cache to True
            sources :
                - module: "MNIST"
                  origin: "torchvision.datasets"
                  kwargs:
                    root: "./MNIST"
                    train: true
                    download: true
```

#### Use the SourceWrapper
Directly using the PyTorch source lead Deeplodocus to choose some parameters by default (is the data loaded ? is the data transformed ? What is the number of instances within the source ? ...)
Deeplodocus offers a SourceWrapper you can use in order to select all the parameters manually.

 is_loaded: bool = True,
                 is_transformed: bool = False,
                 num_instances: Optional[int] = None,
                 instance_id: int = 0,
                 instance_indices: Optional[List[int]] = None):

```yaml
            sources :
                  module: "SourceWrapper"
                  origin: Null
                  kwargs:
                      module: "MNIST"
                      origin: "torchvision.datasets"
                      kwargs:
                        root: "./MNIST"
                        train: true
                        download: true
```
This will give exactly the same result as loading MNIST directly.
Additionally, you can give other arguments:

`is_loaded` (bool), default: true: Whether or not the data need to be loaded by Deeplodocus

`is_transformed` (bool), default: false: Whether or not the data need to be transformed by Deeplodocus

`num_instances` (Optional int), default : Null : Number of raw instances within the Source

`instance_id` (int), default : 0: Index of the instance we want to access


## Create a custom Source

If you want to create de custom Deeplodocus Source, please refer to the [CREATING MODULES](creating_modules.md#Data_Sources) section

## Incoming Sources

In a very near future, Deeplodocus will accept two type of Sources :

#### Loadable Sources

A LoadableSource instance is an object allowing to load data from a source and to pre-store it into memory in order to reduce the loading time during the training.

#### Unlimited Sources

UnlimitedSource instances are objects allowing to load data from sources whose the data stream is unlimited (e.g. a camera or a 3d environment)