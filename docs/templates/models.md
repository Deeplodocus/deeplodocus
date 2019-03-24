# Models

Deeplodocus comes packaged with some pre-defined neural network architectures:
- LeNet
- AlexNet
- VGG
- Darknet-53
- YOLOv3

Use entries in the config/model.yaml file to specify the model you wish to load (see Config.md for more details).

Deeplodocus in-house models can be found in deeplodocus.app.models 

## LeNet

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

## AlexNet

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

## VGG

### Classifiers

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
- **num_classes**: (int) Number of output channels
- **batch_norm**: (bool) include batch normalization after each convolutional layer
- **pretraied**: (bool) initialise the network with weights learned from training with ImageNet
- **include_top**: (bool) include the classifier portion of the network 

### Decoders

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


### AutoEncoder

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
- **decoder:** (dict) specify the decoder module

**NB:** encoder.kwargs.include_top must be set to False. 

## Darknet-53

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

- **num_channels:** (int) specify the number of channels in the input images.

- **include_top:** (bool) specify whether the network should be define with or without its fully-connected classifying layer.

- **num_classes:** (int) [only used of include_top is True] specify the number of output classes at the output of the fully-connected classifier layer.

**NB**: Darknet53 does not require the specification of input height or width, (due to the use of global pooling after convolution).
Inputs are downsampled by a factor of 32, therefore input height and width must each be a multiple of 32. 

## Darknet-19 (COMING SOON)

Original paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf

~~~python
Darknet19(num_channels=3, include_top=True, num_classes=80)
~~~

Config Example: 
~~~yaml
name: Darknet19                             
module: deeplodocus.app.models.darknet 

# Default kwargs     
kwargs:                                     
  num_channels: 3
  include_top: True
  num_classes: 80
~~~

- **num_channels:** (int) specify the number of channels in the input images.
- **include_top:** (bool) specify whether the network should be define with or without its fully-connected classifying layer.
- **num_classes:** (int) [only used of include_top is True] specify the number of output classes at the output of the fully-connected classifier layer.

## YOLO v3

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
      num_channels: 1
      include_top: False
  num_classes: 80
  image_shape: [256, 256]
~~~


- **backbone:** (dict) specify the backbone architecture
    - name: (str): Name of the PyTorch model (nn.Module)
    - module: (str): Location of the module
    - kwargs: (dict):
        - (These depend on the backbone specified)
- **include_top:** (bool) specify whether the network should be define with or without its fully-connected classifying layer
- **num_classes:** (int) specify the number of unique object classes
- **image_shape:** (list of ints) specify the input shape of the image (width, height).