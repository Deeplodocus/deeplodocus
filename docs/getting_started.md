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
