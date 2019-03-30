# Models

Deeplodocus enables to flexibly load and train your own Pytorch neural networks, or one of the pre-defined models that comes packaged with Deeplodocus. 

Specify the neural network that you wish to use via the `name` and `module` entries in the `model.yaml` configuration file.  

- Use `name` to specify the Pytorch nn.Module you would like to initialise.
- Use `module` to specify the Python module to load from. 

More importantly, any custom network architectures defined in the `modules.models` directory of your project can also be easily.

For example:

If you include a python module `skynet.py` in your Deeplodocus project `modules.model` directory, which defines your own artificial neural network, `SkyNet`, you can access your file via the following `model.yaml` configurations: 

defined your own network architecture, SkyNet, inside a python file named skynet.py.
You would write your model configuration file like so: 

```yaml
name: "SkyNet"
module: "modules.models.skynet"
```

NB: If `module` is left empty (Null), Deeplodocus will search through both `deeplodocsus.app.models` and `modules.models` from your project.
The user will be notified if multiple models with the same name are found, and asked to decide which should be used. 


## Pre-defined Models

Deeplodocus comes packaged with some pre-defined neural network architectures:
- LeNet
- AlexNet
- VGG
- Darknet-53
- YOLOv3

Deeplodocus in-house models can be found in deeplodocus.app.models. 

### LeNet

Source: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

Python example:
~~~python
LeNet(num_channels=1, num_classes=10)
~~~

Config example: 
~~~yaml
name: LeNet
module: deeplodocus.app.models.lenet
input_size: [1, 32, 32]
kwargs:                                     
  num_channels: 1
  num_classes: 1000
~~~

- **num_channels**: (int) Number of input channels
- **num_classes**: (int) Number of output channels

### AlexNet

Source: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf

Python example:
~~~python
AlexNet(num_channels=3, num_classes=1000)
~~~

Config example: 
~~~yaml
name: LeNet
module: deeplodocus.app.models.alexnet
input_size: [1, 224, 224]
kwargs:                                     
  num_channels: 3
  num_classes: 1000
~~~

Keyword arguments: 
- **num_channels**: (int) Number of input channels
- **num_classes**: (int) Number of output channels

### VGG

#### Classifiers

Source: https://arxiv.org/pdf/1409.1556.pdf

~~~python
VGG11(num_channels=3, num_classes=1000, batch_norm=True, pre_trained=False, include_top=True)
VGG13(num_channels=3, num_classes=1000, batch_norm=True, pre_trained=False, include_top=True)
VGG16(num_channels=3, num_classes=1000, batch_norm=True, pre_trained=False, include_top=True)
VGG19(num_channels=3, num_classes=1000, batch_norm=True, pre_trained=False, include_top=True)
~~~

Config example: 
~~~yaml
name: VGG16
module: deeplodocus.app.models.vgg
input_size: [1, 224, 224]
kwargs:                                     
  num_channels: 3
  num_classes: 1000
  batch_norm: True
  pre_trained: False
  include_top: True
~~~

Keyword Arguments: 
- **num_channels**: (int) Number of input channels
- **num_classes**: (int) [only used of include_top is True] Number of output channels
- **batch_norm**: (bool) include batch normalization after each convolutional layer
- **pretraied**: (bool) initialise the network with weights learned from training with ImageNet
- **include_top**: (bool) include the classifier portion of the network 

#### Decoders

VGG convolutional layers in reverse order, with torch.nn.Upsample instead of torch.nn.MaxPool2d.

Python example:
~~~python
VGG11Decode(num_classes=10, batch_norm=True)
VGG13Decode(num_classes=10, batch_norm=True)
VGG16Decode(num_classes=10, batch_norm=True)
VGG19Decode(num_classes=10, batch_norm=True)
~~~

Config example: 
~~~yaml
name: VGG16
module: deeplodocus.app.models.vgg
input_size: [512, 8, 8]
kwargs:                                     
  num_classes: 10
  batch_norm: True
~~~

Keyword Arguments: 
- **num_classes**: (int) Number of output channels
- **batch_norm**: (bool) include batch normalization after each convolutional layer


#### AutoEncoder

Auto-encoder using VGG feature detectors. 

~~~python
AutoEncoder(
    encoder={
        "name": "VGG16",
        "module": "deeplodocus.app.models.vgg",
        "kwargs": {
            "batch_norm": True,
            "pretrained": True,
            "num_channels": 3,
            "include_top": False
        } 
    }, 
    decoder={
        "name": "VGG16",
        "module": "deeplodocus.app.models.vgg",
        "kwargs": {
            "num_classes": 10,
            "batch_norm": True
        } 
    )
~~~

Config example: 
~~~yaml
name: AutoEncoder
module: deeplodocus.app.models.vgg
input_size: [3, 256, 256]
kwargs:                                     
    encoder:
    name: VGG16
    kwargs:
      num_channels: 3
      batch_norm: True
      pretrained: False
      include_top: False
  decoder:
    name: VGG16Decode
    kwargs:
      num_classes: 10
      batch_norm: True
~~~

Keyword Arguments: 
- **encoder:** (dict) specify the encoder module
    - **name:** (str) name of the pytorch nn.Module to load
    - **module:** (str) path to the python module to load from
    - **kwargs:** (dict) this depends on encoder module selected
- **decoder:** (dict) specify the decoder module
    - **name:** (str) name of the pytorch nn.Module to load
    - **module:** (str) path to the python module to load from
    - **kwargs:** (dict) this depends on the decoder module selected

**NB:** if using a VGG encoder, encoder.kwargs.include_top must be set to False. 

### Darknet-53

Source: https://pjreddie.com/media/files/papers/YOLOv3.pdf

Python example: 
~~~python
Darknet53(num_channels=3, include_top=True, num_classes=80)
~~~

Config example:
~~~yaml
name: Darknet53
module: deeplodocus.app.models.darknet
input_shape: [3, 256, 256]
kwargs:                                     
  num_channels: 3
  include_top: True
  num_classes: 80
~~~

Keyword arguments:
- **num_channels:** (int) specify the number of channels in the input images.
- **include_top:** (bool) specify whether the network should be define with or without its fully-connected classifying layer.
- **num_classes:** (int) [only used of include_top is True] specify the number of output classes at the output of the fully-connected classifier layer.

**NB**: Darknet53 does not require the specification of input height or width, (due to the use of global pooling after convolution).
Inputs are downsampled by a factor of 32, therefore input height and width must each be a multiple of 32. 

### Darknet-19 (COMING SOON)

Original paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf

~~~python
Darknet19(num_channels=3, include_top=True, num_classes=80)
~~~

Config Example: 
~~~yaml
name: Darknet19                             
module: deeplodocus.app.models.darknet 
input_shape: [3, 256, 256]   
kwargs:                                     
  num_channels: 3
  include_top: True
  num_classes: 80
~~~

- **num_channels:** (int) specify the number of channels in the input images.
- **include_top:** (bool) specify whether the network should be define with or without its fully-connected classifying layer.
- **num_classes:** (int) [only used of include_top is True] specify the number of output classes at the output of the fully-connected classifier layer.

### YOLO v3

Original Paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf

Python example:
~~~python
YOLOv3(
    backbone={
        name: "Darknet53",
        module: "deeplodocus.app.models.darknet",
        kwargs: {
            num_channels: 3,
            include_top: False}
        }, 
    num_classes=80, 
    skip_layers=(36, 61), 
    num_anchors=3, 
    image_shape=(256, 256)
)

~~~
Config example: 
~~~yaml
name: YOLOv3
module: deeplodocus.app.models.yolo:
input_shape: [3, 256, 256]
kwargs:
  backbone:
    name: Darknet53
    module: deeplodocus.app.models.darknet 
    kwargs: 
      num_channels: 3
      include_top: False
  num_classes: 80
  image_shape: [256, 256]
~~~

Keyword arguments: 
- **backbone:** (dict) specify the backbone architecture
    - **name:** (str): name of the pytorch nn.Module to load
    - **module:** (str): path to the python module to load from
    - **kwargs:** (dict) this depends on the backbone selected
- **num_classes:** (int) the number of output classes
- **image_shape:** (list of ints) shape of the input image (width, height).