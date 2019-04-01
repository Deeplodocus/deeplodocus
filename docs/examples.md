## Example # 1: MNIST

In this example, we focus on training the [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) architecture to classify handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

1. **Initialising the project**

    Firstly, we need to create a Deeplodocus project for MNIST with:

    ```
    deeplodocus startproject MNIST
    ```
    
2. **Preparing the dataset**

    Place the MNIST dataset into `MNIST/data`.
    
    Your data directory should look like this:
    
    ```
    data
        ⊦ train
        |   ⊦ images        # Directory of all training images
        |   ∟ labels.dat    # File of all training labels
        ∟ test
            ⊦ images        # Directory of all test images
            ∟ labels.dat    # File of all test labels
    ```

3. **Data configurations**

    Now that we have everything that we need, we can set our project configurations, starting with `config/data.yaml`:
 
    1. Select a batch of 32 images by setting
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

4. **Transforms**

    Inside `config/transform.yaml`, we can assign a transforms file to each of the inputs, labels and additional data for each sub-set of our data. 
    
    By default, transforms for the training and validation input data are specified in `config/transforms/transform_input.yaml`, and in `transform_label.yaml` for the label data. 
    However, in this example we do not need to apply any transforms to our labels, thus `config/transform.yaml` is as follows: 

    ```yaml
    train:
      name: "Train Transform Manager"
      inputs: "config/transforms/transform_input.yaml"
      labels: Null
      additional_data: Null
    validation:
      name: "Validation Transform Manager"
      inputs: "config/transforms/transform_input.yaml"
      labels: Null
      additional_data: Null
    ```
    
    **NB:** test and predict settings are unused and not displayed here.
    
    Now, let's prescribe some input transforms by editing `config/transforms/transform_input.yaml`:

    ```yaml
    method: "sequential"            # Type of transformer
    name: "Input transform"         # Name of the transformer
    
    mandatory_transforms_start: Null # List all the transforms to operate at start
    
    
    transforms:                     # List all the transforms to operate
    
      - name: "normalize_image"     # Normalize the image
        module: Null                # The normalize_image module will be searched automatically by Deeplodocus
        kwargs:
          mean: 127.5
          standard_deviation: 255
    
    mandatory_transforms_end: Null        # List all the transforms to operate at the end
    ```

    **NB:** This transformer is a [Sequential Transformer](deeplodocus.org/en/master/transformer#sequential). 
    Future versions of deeplodocus will remove the mandatory transforms which are not necessary for this particular type of transformer.

5. **The Neural Network**

    Now that our data is ready and will be transformed as we need, we can specify our model in `config/model.yaml`.
    
    We can import a pre-defined [LeNet](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) from `deeplodocus.app.models.lenet` and set the number of classes as a keyword argument as follows:

    ```yaml
    module: "deeplodocus.app.models.lenet"
    name: "LeNet"
    input_size:
      - [1, 28, 28]
    kwargs: 
      num_classes: 10 
    ```
    
    **NB:** test and predict settings are unused and not displayed here.

6. **Optimizer**

    Next, we define the optimizer in `config/optimizer.yaml`.
    
    Below, we import the `Adam` optimizer directly from `torch.optim` and use the default parameters:
    
    ```yaml
    # config/optimizer.yaml
    
    module: "torch.optim"   # Load a predefined optimizer from PyTorch
    name: "Adam"            # Load Adam
    kwargs: {}              # No arguments => Use the default parameters
    ```
    
7. **Training**
    
    Finally, the stage is set to train our network. 
    
    Simply run `main.py` in the project directory to start Deeplodocus, and use the following commands to begin training:
    
    - `load()`
    - `train()`
    
    Once training has finished, you can plot the training history with:
    
    - `plot_history(one_loss=True)`

## Example # 2: CityScapes

In this example, we focus on training a VGG auto-encoder with the [CityScapes](https://www.cityscapes-dataset.com/) dataset. 

The CityScapes dataset was designed for developing semantic understanding of urban street scenes, and is often associated with autonomous driving applications. 

1. **Initialising the project**

    Firstly, we need to create a Deeplodocus project for MNIST with:

    ```
    deeplodocus startproject CityScapes
    ```
    
2. **Preparing the dataset**

    Place the CityScapes dataset into `CityScapes/data`.
    
    Your data directory should look like this:
    
    ```
    data
        ⊦ train
        |    ⊦ images        # Directory of all training images
        |    ∟ labels        # Directory of all label images
        ⊦ val
        |    ⊦ images        # Directory of all test images
        |    ∟ labels        # Directory of all label images
        ∟ ...
    ```
