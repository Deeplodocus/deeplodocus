# Tips & Tricks

## Config Snapshots

During a project, you may wish to save snapshots of your project configurations for later reference. 
This can be done with:

```python
config.snapshot()
```

This will write all current configurations into a single, timestamped file in the project session directory.

Alternatively, to save current configurations to any path, use: 

```python
config.save(file_path="my_path.yaml")
```

## Adjusting Hyper Parameters

Project configurations can be accessed and changed directly from the Deeplodocus terminal.
When combine with the `on_wake` feature, this enables training parameters to be changed during easily and automatically during the training procedure.  

For example, if we wanted to alter the optimizer learning-rate to 0.02 before continuing training, we could specify an `on_wake` command as follows: 

```yaml
# Training, on a pre-define learning-rate for a pre-defined number of epochs
# Then changing the learning-rate and re-loading the optimiser before continuing training
on_wake:
  - load()
  - train()
  - config.optimizer.kwargs.lr = 0.02
  - load_optimizer()      # The optimizer must be re-loaded after the learning-rate has been reset
  - train()
  - sleep()
```

In addition, one could also alter the number of epochs between learning-rate changes, however be sure to re-load the trainer with `load_trainer()`, otherwise the changes would be manifested. 