## Output Transforms

During the training, validation and testing pipelines, illustrated below, output transformations are applied to the output from the model, and the result is parsed into each of the metrics. 
 
![figure](https://github.com/Deeplodocus/deeplodocus/blob/master/docs/figures/ot-1.png)

**Pre-process, Analyse, Visualise:**

The principal reason for output transforms is to enable use of pre-processing algorithms, such as non-maximum suppression, before the evaluation of metrics. This further modularises the deep learning pipeline and reduces the re-computation of identical routines in cases of multiple metrics.

The secondary reason for output transforms is to enable the computation of more complex performance analysis that cannot be encoded into a single metric value. For example, because a confusion matrix cannot be reported as a single value it is not well suited for implementation as a metric in Deeplodocus, however a this type of analysis can still be evaluated and reported through the output transforms. 

An additional benefit of output transforms the opportunity to get creative. Other creative uses for output transforms include the visualisation, recording and broadcasting of network outputs during training.

**Prediction:**

The prediction pipeline, shown below, differs from other procedures in that there are no losses and metrics to compute.
In this case, output transforms are the final step, and can therefore be a useful means to record, visualise or broadcast network outputs. 

![figure](https://github.com/Deeplodocus/deeplodocus/blob/master/docs/figures/ot-2.png)

#### Implementation

Custom output transforms can be implemented as Python classes or as Python functions.
These can then be included in the Deeplodocus pipeline by reference in the project transform configuration files.

#### Function 

Output transforms can be implemented as Python functions, like so:

```python
# Output transform function example

def output_transform(
    inputs=None,            # Input batch (as seen by the model)
    outputs=None,           # Output batch (from the model)
    labels=None,            # Labels
    additional_data=None,   # Additional Data
    **kwargs                # Any keyword arguments (specified in .yaml file)
    ):              
    
    # Some sort of transformation
    
    return outputs
```

#### Class

Output transforms can be implemented as a Python class, as follows:

```python
# Output transform class example

class OutputTransform(object):

    def __init__(self, **kwargs):
        pass
        
    def forward(self,
        inputs=None,
        outputs=None,
        labels=None,
        additional_data=None
        ):
        
        # Some sort of transformation
        
        return outputs
        
    def finish(self):
        pass
```

Implementing output transformers as Python classes gives rise to multiple advantages, as variables can be stored and recalled in subsequent iterations. To compliment this, transform classes may be defined with a special `finish()` method, which will be called once the last batch of data has been outputted from the model. Thus, any windows that were previously created may be closed, or any analysis that was previously computed may be written to file, etc. 

> **NOTE:**
> 
> The **inputs**, **labels** and **additional_data** keyword arguments are optional and will only be parsed to the transform if required. 

### Configuration

To map output transform functions and classes into your Deeplodocus project, first reference them in an output transformer - a Deeplodocus class that manages a selection of transform methods that are sequentially applied to any given data. Output transformers can be specified in `YAML` files, which define the transformer's name and the sequence of constituent transforms.

Below is an example of how a transformer may be specified to manage two constituent transform methods:

```yaml
# transformer.yaml example

name: Output Transformer
transforms: 
  NMS:                              # Human readable name for the transform. specified by the user
    name: non_maximum_suppression   # The name of the Python class or function
    module: modules.transforms...   # Path to the Python module containing the transform methpd 
    kwargs:                         # Any keyword arguments
      threshold: 0.5
  AnotherTransform:                 # Another transform method...
    name: another_transform
    module: modules.transforms...
    kwargs: {}
```

Once output transformers have been defined, they can be mapped into the training, validation, testing and prediction pipelines through the `config/transform.yaml` file.

The below `transform.yaml` example inserts our above transformer into each of these pipelines:

```yaml
# config/transform.yaml example

train:
  name: Train Transform Manager
  inputs: Null
  labels: Null
  additional_data: Null
  outputs: [transformer.yaml]  # List of training output transformers
validation:
  ...
  outputs: [transformer.yaml]  # List of validation output transformers
test:
  ...
  outputs: [transformer.yaml]  # List of test output transformers
predict:
  ...
  outputs: [transformer.yaml]  # List of prediction output transformers
```

### Applying Transformers to Model Outputs

Given that multiple transformers can be specified and models have multiple outputs, outputs are mapped to transformers in a flexible, but logical fashion.

#### Single Transformer - Single Output

Given a single transformer and a single model output, the transformer is simply applied to the the output, as illustrated below:

![figure](https://github.com/Deeplodocus/deeplodocus/blob/master/docs/figures/st-so.png)

#### Single Transformer - Multiple Outputs

Given a single transformer and a multiple model outputs, the transformer is applied repeatedly to each of the outputs, as illustrated below:

![figure](https://github.com/Deeplodocus/deeplodocus/blob/master/docs/figures/st-mo.png)

#### Multiple Transformers - Single Output

Given a multiple transformers and a single model output, each transformer is applied in series to the output, as illustrated below:

![figure](https://github.com/Deeplodocus/deeplodocus/blob/master/docs/figures/mt-so.png)

#### Multiple Transformers - Multiple Outputs

Given a multiple transformers and multiple model outputs, each transformer is mapped by index to a model output, as illustrated below:

![figure](https://github.com/Deeplodocus/deeplodocus/blob/master/docs/figures/mt-mo.png)

>**NOTE:**
>
> If the number of transformers and number of model outputs are greater than one, then the number of transformers must match the number of outputs.

## Worked Example

TODO