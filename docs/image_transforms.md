# Images

## Generic

### resize

Resize an image to the define dimension.

"nearest" or "linear" can be used as method. If not defined Deeplodocus uses "linear" by default for upsampling and "cubic" for downsampling.

kwargs:

- shape (Tuple, List): The output shape requested
- keep_aspect (bool): (default, False) Whether or not to keep the aspect ratio of the image
- padding (int) (default, 0): The padding to apply
- method: (default, None): The specfic resizing method to use

 ```yaml
# Example of usage of resize
transform:
    - name: "resize"
      module : "deeplodocus.app.transforms.images"
      kwargs:
        shape: [256, 128, 3]
        method: "nearest"
```

### normalize_image

Normalize an image. Works with grayscale and RGB(a) images

kwargs:

- mean (int, float): The mean value (if not given, automatically computed ofr the image)
- standard_deviation (int, float) : The standard deviation (if not given, automatically computed ofr the image)

 ```yaml
# Example of usage of normalize_image on a RGB image
transform:
    - name: "normalize_image"
      module : "deeplodocus.app.transforms.images"
      kwargs:
        mean: [127.5, 127.5, 127.5]
        standard_deviation: 255
```

### rotate

Rotate an image

kwargs:

- angle (float): angle of rotation (anti clock wise)

 ```yaml
# Example of usage of rotate
transform:
    - name: "rotate"
      module : "deeplodocus.app.transforms.images"
      kwargs:
        angle: 53
```

### semi_random_rotate


### random_rotate

## Label and colors

### color2label
Transform an RGB image to a array with the corresponding label indices

kwargs : 
- dict_labels (OrderedDict): An dictionary containing the relation class name => color (e.g. cat : [250, 89, 52])

```yaml
# Example of usage of color2label
transform:
    - name: "color2label"
      module : "deeplodocus.app.transforms.images"
      kwargs:
        dict_labels:
          car : [0, 15, 59]
          pedestrian : [158, 56, 87]
          sidewalk: [87, 63, 22] 
```

### lable2color
Transform an array of labels to the corresponding RGB colors

kwargs : 
- dict_labels (OrderedDict): An dictionary containing the relation class name => color (e.g. cat : [250, 89, 52])

```yaml
# Example of usage of label2color
transform:
    - name: "label2color"
      module : "deeplodocus.app.transforms.images"
      kwargs:
        dict_labels:
          car : [0, 15, 59]
          pedestrian : [158, 56, 87]
          sidewalk: [87, 63, 22] 
```

## Channels transformation

### remove_channel
Remove a channel from the given array

kwargs:
- index_channel (int): The index of the channel to remove
```yaml
# Example of usage of remove channel
transform:
    - name: "remove_channel"
      module : "deeplodocus.app.transforms.images"
      kwargs:
        index_channel: 1  # Remove the second channel
```

### convert_rgba2bgra
Convert a RGB(a) image to BGR(a)

kwargs: None

```yaml
# Example of usage of convert_rgba2bgra
transform:
    - name: "convert_rgba2bgra"
      module : "deeplodocus.app.transforms.images"
      kwargs: Null
```

### convert_bgra2rgba
Convert a BGR(a) image to RGB(a)

kwargs: None

```yaml
# Example of usage of convert_bgra2rgba
transform:
    - name: "convert_bgra2rgba"
      module : "deeplodocus.app.transforms.images"
      kwargs: Null
```

### grayscale
Convert a RGB(a) image to grayscale

kwargs: None
```yaml
# Example of usage of grayscale
transform:
    - name: "grayscale"
      module : "deeplodocus.app.transforms.images"
      kwargs: Null
      
      
```

## Blurs

### bilateral_blur
Apply a bilateral blur to the image

<span style="color:orange">/!\ Require OPENCV to be installed</span>

kwargs:

- diameter (int): Diameter of the kernel
- sigma_color (int): Sigma value of the color space
- sigma_space (int): Sigma value of the coordinate space

For more information, please check [OpenCV documentation on the bilateral filter](https://docs.opencv.org/4.0.1/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed) 

```yaml
# Example of usage of bilateral_blur
transform:
    - name: "bilateral_blur"
      module : "deeplodocus.app.transforms.images"
      kwargs:
        diameter: 3
        sigma_color: 2
        sigma_space: 2
```

### median_blur
Apply a bilateral blur to the image

<span style="color:orange">/!\ Require OPENCV to be installed</span>

kwargs:

- kernel_size (int): Size of the kernel

For more information, please check [OpenCV documentation on the median filter](https://docs.opencv.org/4.0.1/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)

```yaml
# Example of usage of median_blur
transform:
    - name: "median_blur"
      module : "deeplodocus.app.transforms.images"
      kwargs:
        kernel_size: 5
```

### gaussian_blur
Apply a gaussian blur to the image

<span style="color:orange">/!\ Require OPENCV to be installed</span>

kwargs:

- kernel_size (int): Size of the gaussian kernel

For more information, please check [OpenCV documentation on the median filter](https://docs.opencv.org/4.0.1/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)

```yaml
# Example of usage of median_blur
transform:
    - name: "gaussian_blur"
      module : "deeplodocus.app.transforms.images"
      kwargs:
        kernel_size: 5
```

