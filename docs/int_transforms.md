# Integers

## Encoding

### one_hot_encode

Transform an integer to a one hot encoded vector of dimension 1xn with n the number of classes (e.g. n = 5, then the class id 2  becomes [0, 0, 1, 0, 0])

<span style="color:orange">/!\ One hot encode not require if you use the CrossEntropyLoss implemented into PyTorch</span>

kwargs :

- num_classes (int): The number of classes in the output vector

```yaml
# Example of usage of one_hot_encode
transform:
    - name: "one_hot_encode"
      module : "deeplodocus.app.transforms.integers"
      kwargs:
        num_classes: 5
```
