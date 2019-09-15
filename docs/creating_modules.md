# About Custom-Built Modules

Although there are already a great number of useful PyTorch modules that can be mapped into a deep learning pipeline, Deeplodocus also allows you to include your own custom-built modules. The following documentation explains how to implemented and configure your own models, data sources and transforms, losses, metrics and even optimizers.

## Data Sources

#### Definition

A custom Deeplodocus Source is very similar to a PyTorch Dataset. The 

To use already defined Deeplodocus Source classes, please check [EXISTING MODULES](existing_modules/data_sources.md)

#### Configuration

You will have to define the following functions in a class inheriting `Source`:

- \_\_init__(): Initialize the Source instance
- \_\_getitem__(index): Get the items at the desired index
- compute_length(): Return the length of the Source
- (Optional) check(): To make additional checks on your Source when being loaded by Deeplodocus

#### Example

```python
# Python imports
from typing import Tuple, Any, Optional

# Deeplodocus imports
from deeplodocus.data.load.source import Source

class CustomSource(Source):
    def __init__(self,
                 index: int = -1,
                 is_loaded: bool = False,
                 is_transformed: bool = False,
                 num_instances: Optional[int] = None,
                 instance_id: int = 0
                 ):

        super().__init__(index=index,
                         num_instances=num_instances,
                         is_loaded=is_loaded,
                         is_transformed=is_transformed,
                         instance_id=instance_id)


    def __getitem__(self, index: int) -> Tuple[Any, bool, bool]:

        # Get is loaded and transformed
        is_loaded = self.is_loaded
        is_transformed = self.is_transformed
        
        # Function to load the data anyhow
        data = get_custom_data()
        
        # Return the data, is_loaded and is_transformed
        return data, is_loaded, is_transformed

    def compute_length(self) -> int:
        """
        Custom function to compute the length of the Source"""
        length = 0
        
        return length

    def check(self) -> None:
        """
        Custom function used to check the correctness of the Source"""
        # Perform the super().check() first
        super().check()
        
        # Perform any custom check


```
## Models

### Definition

Deeplodocus models follows the PyTorch conventions. 
### Configuration

All you will need are a class inheriting `torch.nn.Module`, an \_\_init__() method and a forward() method.
### Example

The following example is used to create LeNet for MNIST.
```python
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self, num_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 84)
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
```
## Losses

#### Definition

TODO: how to define a custom loss

#### Configuration

TODO: how to configure a custom loss

#### Example

TODO: example of implementing a custom loss

## Metrics

#### Definition

TODO: how to define a custom metric

#### Configuration

TODO: how to configure a custom metric

#### Example

TODO: example of implementing a custom metric

## Transforms

### Data Transforms

#### Definition

TODO: how to define a custom data transform

#### Configuration

TODO: how to configure a custom data transform

#### Example

TODO: example of implementing a custom data transform

### Output Transforms

#### Definition

TODO: how to define a custom output transform

#### Configuration

TODO: how to configure a custom output transform

#### Example

TODO: example of implementing a custom output transform

## Optimizers

#### Definition

TODO: how to define a custom optimiser

#### Configuration

TODO: how to configure a custom optimiser

#### Example

TODO: example of implementing a custom optimiser

## Notification

#### Definition

A `Notification` is a message which will be displayed to the user. There are many types of `Notification`, each displayed with a different color:

- DEEP INFO (blue) : Information for the user 
- DEEP RESULT (white): Result during inference time
- DEEP DEBUG (cyan): A debug message
- DEEP WARNING (orange): A warning message
- DEEP SUCCESS (green): A success message
- DEEP ERROR (red): An error message
- DEEP FATAL ERROR (red): An error message stopping the inference
- DEEP LOVE (pink): A message when exiting Deeplodocus
- DEEP INPUT (blinking white): A prompt form

All the `Notification` messages are logged automatically. You can add `Notification` instances to your custom-built modules.

#### Configuration

Any `Notification` contains the following arguments:

- notif_flag (Flag): The type of notification
- message (str): The message to display
- log (bool) default: True: Whether to write the Notification into the logs

For DEEP FATAL ERROR `Notification`, you can add the following argument:
- solutions (List), default: None: The list of solutions to solve the issue

#### Example

```python
from deeplodocus.utils.notification import Notification
from deeplodocus.flags.notif import *

# INFO
Notification(DEEP_NOTIF_INFO, "Info message")

# RESULT
Notification(DEEP_NOTIF_RESULT, "Result message")

# DEBUG
Notification(DEEP_NOTIF_DEBUG, "Debug message")

# WARNING
Notification(DEEP_NOTIF_WARNING, "Warning message")

# SUCCESS
Notification(DEEP_NOTIF_SUCCESS, "Success message")

# ERROR
Notification(DEEP_NOTIF_ERROR, "Error message")

# FATAL ERROR
Notification(DEEP_NOTIF_FATAL, "Fatal error message", 
                solutions=["You can do this to solve the issue", "You can do that to solve the issue"])
```