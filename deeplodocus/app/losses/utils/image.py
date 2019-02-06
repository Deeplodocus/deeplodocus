import torch.nn.functional as F



def gradient_x(img):
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]
    return gx

def gradient_y(img):
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]
    return gy


def pyramidal_images(image, n=4):
    """
    AUTHORS:
    --------

    :author: Alix Leroy


    DESCRIPTION:
    ------------

    Generate a pyramid of sub images.
    For each nth image with (c, h, w) dimensions, the (n+1)th image has (c, h/2, w/2) dimensions


    PARAMETERS:
    -----------

    :param image (Tensor): The original image
    :param n: Number of total images required (The original image is included in the total images)

    RETURN:
    -------

    :return pyramid(List[Tensor]): The pyramid
    """
    pyramid = []
    for i in range(n):
        img = F.interpolate(image, scale_factor=1/(2**i))
        pyramid.append(img)
    return pyramid