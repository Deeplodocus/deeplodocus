# Images

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

For more information, please check [OpenCV documentation on the median filter](    https://docs.opencv.org/4.0.1/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)

```yaml
# Example of usage of median_blur
transform:
    - name: "median_blur"
      module : "deeplodocus.app.transforms.images"
      kwargs:
        kernel_size: 5
```

