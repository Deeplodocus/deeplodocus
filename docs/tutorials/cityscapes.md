

# Cityscapes with SegNet

#### Define the Data
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
    - name: "Train Cityscapes"
      type: "train"
      num_instances: 60000
        entries:
            -   name: "Cityscapes image"
                type: "input"
                load_as: "image"
                convert_to: "float32"
                move_axis: [2, 1, 0]
                enable_cache: true
                sources :
                    - name: "Cityscapes"
                      module: "torchvision.datasets"
                      kwargs:
                        root: "./data/cityscapes"
                        split: "train"
                        target_type: "semantic"
    
            -   name: "Cityscapes semantic"
                type: "label"
                load_as: "image"
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
    - name: "Validate Cityscapes"
      type: "validation"
      num_instances: Null
        entries:
            -   name: "Cityscapes image"
                type: "input"
                load_as: "image"
                convert_to: "float32"
                move_axis: [2, 1, 0]
                enable_cache: true
                sources :
                    - name: "MNIST"
                      module: "torchvision.datasets"
                      kwargs:
                        root: "./data/cityscapes"
                        split: "val"
                        mode: "coarse"
                        target_type: "semantic"
    
            -   name: "Cityscapes semantic"
                type: "label"
                load_as: "image"
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

#### Define the Data Transformations

Transform the input images:
```yaml
method: "some_of"
name: "Input sequential"

num_transformations_min: 0
num_transformations_max: 1

mandatory_transforms_start:

  # We resize the image to [256, 128, 3], the expected input size of the network
  - name: resize
    module: 
    kwargs:
      shape: [256, 128, 3]
      padding: 0
      keep_aspect: True
      
    - randomcrop:
      name: "random_crop"
      module: Null
      kwargs:{}
      
  - randomintensityshit
  - blurs      

transforms:
    - name : "random_blur"
      module:
      kwargs:
        kernel_size_min: 3
        kernel_size_max : 5

mandatory_transforms_end:
  # Normalize the image
  - name: "normalize_image"
    module:
    kwargs:
      mean: [128.0, 128.0, 128.0]
      standard_deviation: 255
```

Tranform the labels:

```yaml
method: "sequential"
name: "Label sequential"

mandatory_transforms_start: Null

transforms:

  # We first remove the last channel
  # The label image is an RGB(a) image. However, no image is encoded in the alpha channel (index 3)
  - rgba2rgb:
    name: remove_channel
    module:
    kwargs:
      index_channel: 3
      

  # We then resize the image to [256, 128, 3], the expected input size of the network
  - resize:
    name: resize
    module: 
    kwargs:
      shape: [256, 128, 3]
      method: "nearest"
      
  # Images are encoded with RGB colors.
  # We want to convert the color to a specific index corresponding to a class
  # This function is an example, it is not optimal for large images and therefore is very slow
  # A Cython function could be considered to speed up the python loops
  # We give a dictionary as input containing all the class names and there corresponding colors
  - color2label:
    name: color2label
    module:
    kwargs: 
      dict_labels: 
        unlabeled: [0, 0, 0]
        dynamic: [111, 74, 0]
        ground: [81, 0, 81]
        road: [128, 64,128]
        sidewalk: [244, 35,232]
        parking: [250,170,160]
        rail track: [230,150,140]
        building: [70, 70, 70]
        wall: [102,102,156]
        fence: [190,153,153]
        guard rail: [180,165,180]
        bridge: [150,100,100]
        tunnel: [150,120, 90]
        pole: [153,153,153]
        polegroup: [153,153,153]
        traffic light: [250,170, 30]
        traffic sign: [220,220, 0]
        vegetation: [107,142, 35]
        terrain: [152,251,152]
        sky: [70,130,180]
        person: [220, 20, 60]
        rider: [255, 0, 0]
        car: [0, 0,142]
        truck: [0, 0, 70]
        bus: [0, 60,100]
        caravan: [0, 0, 90]
        trailer: [0, 0,110]
        train: [0, 80,100]
        motorcycle: [0, 0,230]
        bicycle: [119, 11, 32]
        license plate: [0, 0,142] 
  # The output array has the following shape:  [256, 128, 1]
  # We want to reshape it to [256, 128]
  - reshape:
    name: reshape
    module:
    kwargs:
      shape: [256, 128]

mandatory_transforms_end: Null
```
#### Define the Model

```yaml
module: deeplodocus.app.models.vgg
name: SegNet
file: Null
from_file: False
input_size:
  - [3, 128, 256]
kwargs:

  # SegNet is divided into two subnetworks
  # 1) The encoder (here based on VGG11) taking 3-channels input images
  # 2) The decoder (here based on VGG11 with 31 output classes)
  
  # Encoder (VGG11)
  encoder:
    name: VGG11
    module: deeplodocus.app.models.vgg
    kwargs:
      batch_norm: True
      num_channels: 3 # RGB images as input
      include_top: False
  
  # Decoder (VGG11)
  decoder:
    name: VGG11Decode
    module: deeplodocus.app.models.vgg
    kwargs:
      batch_norm: True
      num_classes: 31 # 31 classes as output
```

#### Define the Loss function

```yaml
# losses.yaml
CrossEntropy:
  module: Null
  name: "CrossEntropy2d"
  weight: 1
  kwargs:
```

#### Metrics, Optimizer and Project

The metrics, optimizer and project files will be the same as the MNIST tutorial. :)
