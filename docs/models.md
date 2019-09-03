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

Source: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

#### Python example

~~~python
LeNet(num_channels=1, num_classes=10)
~~~

#### Config example

~~~yaml
name: LeNet
module: deeplodocus.app.models.lenet
input_size: [1, 32, 32]
kwargs:                                     
  num_channels: 1
  num_classes: 1000
~~~

#### Keyword arguments

- **num_channels**: (int) Number of input channels
- **num_classes**: (int) Number of output channels

#### Return

Tensor of size (batch size x num classes)

### AlexNet

Source: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

#### Python example

~~~python
AlexNet(num_channels=3, num_classes=1000)
~~~

#### Config example

~~~yaml
name: LeNet
module: deeplodocus.app.models.alexnet
input_size: [1, 224, 224]
kwargs:                                     
  num_channels: 3
  num_classes: 1000
~~~

#### Keyword arguments 

- **num_channels**: (int) Number of input channels
- **num_classes**: (int) Number of output channels

#### Return

Tensor of size (batch size x num classes).

### VGG

#### Classifiers

Source: [Very Deep Convolutional Networks For Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)

#### Python example

~~~python
VGG11(num_channels=3, num_classes=1000, batch_norm=True, pre_trained=False, include_top=True)
VGG13(num_channels=3, num_classes=1000, batch_norm=True, pre_trained=False, include_top=True)
VGG16(num_channels=3, num_classes=1000, batch_norm=True, pre_trained=False, include_top=True)
VGG19(num_channels=3, num_classes=1000, batch_norm=True, pre_trained=False, include_top=True)
~~~

#### Config example

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

#### Keyword Arguments

- **num_channels**: (int) Number of input channels
- **num_classes**: (int) [only used of include_top is True] Number of output channels
- **batch_norm**: (bool) include batch normalization after each convolutional layer
- **pretraied**: (bool) initialise the network with weights learned from training with ImageNet
- **include_top**: (bool) include the classifier portion of the network 

#### Return

- If `include_top` is `True`: tensor of size (batch size x num classes)
- If `include_top` is `False`: tensor of size (batch size x h x w x num features)

#### Decoders

VGG convolutional layers in reverse order, with torch.nn.Upsample instead of torch.nn.MaxPool2d.

#### Python example

~~~python
VGG11Decode(num_classes=10, batch_norm=True)
VGG13Decode(num_classes=10, batch_norm=True)
VGG16Decode(num_classes=10, batch_norm=True)
VGG19Decode(num_classes=10, batch_norm=True)
~~~

#### Config example

~~~yaml
name: VGG16
module: deeplodocus.app.models.vgg
input_size: [512, 8, 8]
kwargs:                                     
  num_classes: 10
  batch_norm: True
~~~

#### Keyword Arguments

- **num_classes**: (int) Number of output channels
- **batch_norm**: (bool) include batch normalization after each convolutional layer

#### Return

Tensor of size (batch size x h x w x num features)

#### AutoEncoder

Auto-encoder using VGG feature detectors. 
A VGG decoder with softmax classifier is directly appended to a VGG encoder. 

#### Python example

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

#### Config example

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

#### Keyword arguments

- **encoder:** (dict) specify the encoder module
    - **name:** (str) name of the pytorch nn.Module to load
    - **module:** (str) path to the python module to load from
    - **kwargs:** (dict) this depends on encoder module selected
- **decoder:** (dict) specify the decoder module
    - **name:** (str) name of the pytorch nn.Module to load
    - **module:** (str) path to the python module to load from
    - **kwargs:** (dict) this depends on the decoder module selected

#### Return

Tensor of size (batch size x h x w x num classes)

**NB:** if using a VGG encoder, encoder.kwargs.include_top must be set to False. 

### Darknet-53

Source: [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

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

#### Keyword arguments

- **num_channels:** (int) specify the number of channels in the input images.
- **include_top:** (bool) specify whether the network should be define with or without its fully-connected classifying layer.
- **num_classes:** (int) [only used of include_top is True] specify the number of output classes at the output of the fully-connected classifier layer.

**NB:** 
- Inputs are downsampled by a factor of 32, therefore input height and width must each be a multiple of 32.
- Outputs of layers 36 and 61 are stored in a dictionary, `Darknet53.skip[36]` and Darknet53.skip[61]` respectively (for use with YOLO).

### Darknet-19 (COMING SOON)

Source: [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)


#### Python example

~~~python
Darknet19(num_channels=3, include_top=True, num_classes=80)
~~~

#### Config example

~~~yaml
name: Darknet19                             
module: deeplodocus.app.models.darknet 
input_shape: [3, 256, 256]   
kwargs:                                     
  num_channels: 3
  include_top: True
  num_classes: 80
~~~

#### Keyword arguments

- **num_channels:** (int) specify the number of channels in the input images.
- **include_top:** (bool) specify whether the network should be define with or without its fully-connected classifying layer.
- **num_classes:** (int) [only used of include_top is True] specify the number of output classes at the output of the fully-connected classifier layer.

#### Return

- If `include_top` is `True`: tensor of size (batch size, num classes)
- If `include_top` is `False`: tensor of size (batch size, h, w, num features)

### YOLO v3

Source: [You Only Look Once: Unified, Real-Time Object Detection](https://pjreddie.com/media/files/papers/yolo_1.pdf)

Source: [YOLO9000: Faster, Better, Stronger](https://pjreddie.com/media/files/papers/YOLO9000.pdf)

Source: [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

#### Python example

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
    input_shape=(256, 256),
    anchors=(
        ((116, 90), (156, 198), (373, 326)),
        ((30, 61), (62, 45), (59, 119)),
        ((10, 13), (16, 30), (22, 23))
    ),
    normalized_anchors=False
)
~~~

#### Config example

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
  skip_layers: [36, 61]
  image_shape: [256, 256]
  anchors: 
    - [[116, 90], [156, 198], [373, 326]]
    - [[30, 61], [62, 45], [59, 119]]
    - [[10, 13], [16, 30], [22, 23]]
  normalized_anchors: False
  predict: False
~~~

#### Keyword arguments

- **backbone:** (dict) specify the backbone architecture
    - **name:** (str): name of the pytorch nn.Module to load
    - **module:** (str): path to the python module to load from
    - **kwargs:** (dict) this depends on the backbone selected
- **num_classes:** (int) the number of output classes
- **skip_layers:** (list of ints) the backbone layers to skip connect with
- **image_shape:** (list of ints) shape of the input image (width, height)
- **anchors:** (list of lists of ints) anchor box (a.k.a. priors) width and height values for each of the three detection layers
- **normalized_anchors:** (bool) whether the given anchors are normalized or not (If True, anchor values should be between 0 and 1)
- **predict:** (bool) whether or not to set the model to predict mode

#### Return

- If in **train** and **eval** mode: tuple of prediction and scaled_anchors for each detection layer.
- If in **predict** mode: tensor of predictions (batch size x num predictions x (num classes + 5)). In the the third dimension, values are: box x coordinate, box y coordinate, box width, box height and objectness score followed by class scores (x, y, w, h, obj, cls).

**NB:** Before using any prediction functionality, YOLO should be set to predict mode. This can be done by entering `model.predict()` in the Deeplodocus terminal.