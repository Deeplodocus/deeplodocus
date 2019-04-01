## Example # 1: MNIST

In this example, we focus on training the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) architecture to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

1. Firstly, we need to create a Deeplodocus project with:

    ```text
    deeplodocus startproject MNIST
    ```
2. Place the MNIST dataset into `MNIST/data`.
    Your data directory should look like this:
    ```text
    data
       ⊦ train
       |   ⊦ images        # Directory of all training images
       |   ∟ labels.dat    # File of all training labels
       ∟ test
           ⊦ images        # Directory of all test images
           ∟ labels.dat    # File of all test labels
    ```

3. Now that we have everything that we need, we can set our project configurations, starting with `config/data.yaml`:
 
    1. Select a batch of 32 images
    2. Select 4 workers (processes) to load the images (hardware dependant)
    3. Enable training by setting `enabled/train: True`
    4. Enable validation by setting `enabled/validation: True`
    5. Define 1 input located into the folder (the handwritten digit images)
    6. Define 1 label (the label corresponding to the image)

    The `config/data.yaml` looks as below :

    ```yaml
    # config/data.yaml
    
    dataloader:
      batch_size: 32              # 32 images per batch
      num_workers: 4              # 4 Processes
      
    enabled:
      train: True                 # Enable The training step
      validation: True            # Enable the validation step
      test: False
      predict: False
      
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

    **NB:** test and predict settings are unused and not displayed here.

4. Next, we can assign some data transforms in `config/transform.yaml`:

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