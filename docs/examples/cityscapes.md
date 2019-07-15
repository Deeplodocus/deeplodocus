## Example : CityScapes

In this example, we focus on training a VGG auto-encoder with the [CityScapes](https://www.cityscapes-dataset.com/) dataset. 

The CityScapes dataset was designed for developing semantic understanding of urban street scenes, and is often associated with autonomous driving applications. 

![figure](https://github.com/Deeplodocus/deeplodocus/blob/master/docs/figures/cityscapes.png)

1. **Initialising the project**

    Firstly, we need to create a Deeplodocus project for MNIST with:

    ```
    deeplodocus startproject CityScapes
    ```
    
2. **Preparing the dataset**

    Place the CityScapes dataset into `CityScapes/data`.
    
    Your data directory should look like this:
    
    ```txt
    data
        ⊦ train
        |    ⊦ images        # Directory of all training images
        |    ∟ labels        # Directory of all label images
        ⊦ val
        |    ⊦ images        # Directory of all test images
        |    ∟ labels        # Directory of all label images
        ∟ ...
    ```
