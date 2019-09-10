# Starting a Project

Use Deeplodocus's `startproject` command, followed by a name for your new project, to generate a new Deeplodocus directory. This will contain all the files necessary to begin your next deep learning project. Note that `start-project` and `start_project` will also be accepted.

```bash
$ deeplodocus startproject <project-name>
```

Other entry commands include:

- `deeplodocus version` to display the version number of the current Deeplodocus installation.
- `deeplodocus help` to display a list of avalaible entry commands.

## Project Structure

A complete Deeplodocus project directory will contain a main Python script and two sub-directories. 

The first of these is the config directory, which controls different groups of high-level project variables through 9 YAML files. Also within config is a transforms directory, that can be used to house additional YAML files which perscribe routines for transforming input and output data. More detailed information about project configurations can be found on the [Configuration](config.md) page.

The second is a modules directory, which is a place to house any modules that are custom-built for the project. Initially this will be empty, but users can optionally add their own models, data sources and transforms, losses, metrics and optimizers. More detailed information about impleementing and including custom modules can be found on the [Create Modules](creating_modules.md) page.

Once Deeplodocus is executed, another directory will be generated to store any resultant model weight files, logs and training history from the current session. The name of this directory is controlled by the user through the [config/project.yaml](config.md#project) file. 

```python
deeplodocus_project
    ├ config                  # Directory for config files
    │    ├ transforms         # Directory for transformer files
    │    ├ data.yaml
    │    ├ history.yaml
    │    ├ losses.yaml
    │    ├ metrics.yaml
    │    ├ optimzer.yaml
    │    ├ project.yaml
    │    ├ training.yaml
    │    └ transform.yaml
    ├ modules                 # Directory for custom-build modules
    └ main.py                 # Python script for running Deeplodocus
```

# Running a Project

Once the parameters in each of the configuration files have been set, the project is ready to be executed - simply run the main.py scripy with Python in the project directory. Upon running, Deeplodocus will load the project configurations and raise warnings if any settings are missing or invalid - these will be replaced by default values. You will then have access to the Deeplodocus terminal and, if you are satisfied that your project settings have been loaded sucessfully, you can begin to execute commands to activate your deep learning pipeline. A collection of the core Deeplodocus commands are detailed in the [Core Commands](getting_started.md#core-commands) section of this page.

```bash
$ python3 main.py
```

## Core Commands

### Load

```
> load()
```

Upon running a Deeplodocus project the load command will typically be the first to be called, as it loads a series of key Deeplodocus modules. More information about the load command can be found on the [FrontalLobe section of the API](api.md#FrontalLobe) page.


Modules that are initialized by the load command are:

- the model
- the optimizer
- the loss(es)
- any metrics
- the validator (if enabled)
- the tester (if enabled)
- the trainer (if enabled)
- the predictor (if enabled)
- memory (for handling history)

### Train

```
> train()
```

Once all the modules necessary for optimizing the model have been loaded, model training can initiated with the train command. More information about the training process can be found on the [Trainer section of the API page](api.md#trainer).

Note: 

- If a validator has been loaded, at the end of each training epoch the model will be evaluated over the validation dataset.
- If memory has been loaded, training (and validation) history will be saved to the histroy folder within the session directory.
- If any metrics have been loaded, they will be reported alongside loss values. 

Modules that should be loaded before training are:

- the model
- the optimizer
- the loss(es)
- metric(s) (optional)
- the validator (optional)
- the trainer
- memory

### Test

```
> test()
```

Once a model has been trained, the test command can be used to evaluate your model over any test datasets. More information about the test process can be found on the [Tester section of the API page](api.md#tester).

Modules that should be loaded before testing are: 

- the model
- the loss(es)
- metric(s) (optional)
- the tester (optional)
- memory

### Validate

```
> validate()
```

Validation can be done whilst training, however to evaluate your model over the validation dataset seperately, use the validate command. The validator is identical to a tester module, so more information about the validation process can be found on the [Tester section of the API page](api.md#tester).

Modules that should be loaded before validating include: 

- the model
- the loss(es)
- metric(s) (optional)
- the validator (optional)
- memory

### Predict

```
> predict()
```

Prediction is a flexible pipeline for conducting experiments and simulating the deployment of a model. During the prediction, losses and metrics are ignored, and the pipeline terminates with output transforms. More information about the prediction process can be found on the [Predictor section of the API page](api.md#predictor).

Modules that should be loaded before predicting include: 

- the model
- the predictor