# Image Transforms

## Normalize

Normalize an image (works with grayscale, RGB and RGBA images).

#### Python Usage

```python
from deeplodocus.app.transforms.images import normalize_image

np.array = normalize_image(image, mean=, standard_deviation, mean)
```

#### Arguments

- **mean**: (int, float) The mean pixel value of the output image (if not given, automatically computed from the image).
- **standard_deviation**: (int, float) The standard deviation of pixel values in the output image (if not given, automatically computed ofr the image).

#### Deeplodocus Configuration

```yaml
# Example of usage of normalize_image on a RGB image
name: normalize_image
module: deeplodocus.app.transforms.images
kwargs:
  mean: [127.5, 127.5, 127.5]
  standard_deviation: 255
```

## Resize

Resize the given image to the given shape, using the prescribed method.

#### Python Usage

```python
from deeplodocus.app.transforms.images import resize

np.array = resize(image, shape, keep_aspect=False, padding=0, method=None)
```
#### Arguments

- **shape**: (Tuple, List) The output width and height of the image respectively.
- **keep_aspect**: (bool=False) Whether or not to keep the aspect ratio of the image.
- **padding**: (int=0) The padding to apply if keep_aspect is True.
- **method**: (str=None) The specfic resizing method to. Can be one of : "nearest", "linear", "cubic". If methid is None, "linear" will be used when downsampling and "cublic" will be used with upsampling.

#### Deeplodocus Configuration

```yaml
# Example of including resize

name: resize
module : deeplodocus.app.transforms.images
kwargs:
  shape: [256, 128]
  keep_aspect: True
  padding: 0
  method: nearest
```

# Generic Transforms

TODO