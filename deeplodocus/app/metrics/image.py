import torch.nn.functional as F
import torch

def gradient_x(image):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Compute the horizontal gradient of an image

    PARAMETERS:
    -----------

    :param image (Tensor): The original image

    RETURN:
    -------

    :return gx(Tensor): The horizontal gradient
    """
    gx = image[:, :, :, :-1] - image[:, :, :, 1:]
    return gx

def gradient_y(img):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Compute the vertical gradient of an image

    PARAMETERS:
    -----------

    :param image (Tensor): The original image

    RETURN:
    -------

    :return gy(Tensor): The vertical gradient
    """
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



"""
"
" OPTICAL FLOW
"
"""

def absolute_end_point_error(outputs, labels):
    """
    AUTHORS:
    --------

    :author: Alix Leroy


    DESCRIPTION:
    ------------

    Calculate the Absolute End Point Error between two tensors

    PARAMETERS:
    -----------

    :param outputs: Output tensor
    :param labels: Target tensor

    RETURN:
    -------

    :return: The Absolute End Point Error (L2 distance between the two tensors)
    """
    return torch.dist(outputs, labels, 2)


def average_end_point_error(outputs, labels):
    """
    AUTHORS:
    --------

    :author: Alix Leroy


    DESCRIPTION:
    ------------

    Calculate the Average End Point Error between two tensors

    PARAMETERS:
    -----------

    :param outputs: Output tensor
    :param labels: Target tensor

    RETURN:
    -------

    :return: The Average End Point Error
    """
    batch_size, channels, height, width = labels.shape

    # Number of element in an image
    n = batch_size * channels * height * width

    return torch.dist(outputs, labels, 2).div(n)


def average_angular_error(outputs, labels):
    """
    AUTHORS:
    --------

    :author: Alix Leroy


    DESCRIPTION:
    ------------

    Calculate the Average Angular Error between two tensors

    PARAMETERS:
    -----------

    :param outputs: Output tensor
    :param labels: Target tensor

    RETURN:
    -------

    :return: Average Angular Error between two tensors
    """

    # Flat the tensor
    flat_outputs = outputs.view(outputs.numel())
    flat_labels = labels.view(labels.numel())

    dot_product = torch.dot(flat_outputs, flat_labels)

    #    1) AAE = cos^-1(normalized_o, normalized_l)
    # or 2) AAE = atan2(norm(cross(o, l)), dot(o, l))
    return torch.atan2(torch.dist(flat_labels, flat_outputs, 1), dot_product)


"""
"
" DEPTH
"
"""


def supervised_depth_eigen2014a(outputs, labels):
    """
    Loss for depth estimation in supervised learning
    This loss is described in D. Eigen and al. (2014a)
    Title : Depth Map Prediction from a Single Image using a Multi-Scale Deep Network
    Link : https://arxiv.org/abs/1406.2283

    IMPLEMENTATION:
    ---------------

    :implementer: Alix Leroy
    """
    # Size
    batch_size, channels, height, width = labels.shape

    # Number of element in an image
    n = batch_size * channels * height * width

    # Log distance
    d = outputs.log() - labels.log()

    # L2 related term (Note that a pure L2 would use torch.dist() in order to use the squared-root)
    l2 = d.pow(2).sum().div(n)

    # Scale invariant difference : "that credits mistakes if they are in the same direction and penalizes them if they oppose.
    # Thus, an imperfect prediction will have lower error when its mistake sare consistent with one another" (cf. paper)
    SID = d.sum().pow(2).div(2*(n**2))

    # Final loss
    L_depth = l2 - SID

    return L_depth


def supervised_depth_eigen2014b(outputs, labels):
    """
    AUTHORS:
    --------

    :author: Alix Leroy

    DESCRIPTION:
    ------------

    Loss for depth estimation in supervised learning
    D. Eigen and al. (2014b)
    Title : Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture
    Paragraph 4.1 "Depth"
    Link : https://arxiv.org/abs/1411.4734

    PARAMETERS:
    -----------

    :param outputs (Tensor): Output tensors
    :param labels (Tensor): Target tensors

    RETURN:
    -------

    :return L_depth(Tensor): The depth loss
    """
    # Size
    batch_size, channels, height, width = labels.shape

    # Number of element in an image
    n = batch_size * channels * height * width

    # Log distance
    d = outputs.log() - labels.log()
    # d = outputs - labels

    # L2 related term (Note that a pure L2 would use torch.dist() in order to use the squared-root)
    l2 = d.pow(2).sum().div(n)

    # Scale invariant difference : "that credits mistakes if they are in the same direction and penalizes them if they oppose.
    # Thus, an imperfect prediction will have lower error when its mistake sare consistent with one another" (cf. paper)
    SID = d.sum().pow(2).div(2 * (n ** 2))

    # First order matching (compares image gradients of the prediction with the ground truth)
    gx = gradient_x(d)
    gy = gradient_y(d)

    #First Order Matching
    FOM = (gx.pow(2).sum() + gy.pow(2).sum()).div(n)

    # Final loss
    L_depth = l2 - SID + FOM

    return L_depth


def delta_smaller_1_25_p(outputs, labels, p=1):

    v1 = torch.div(outputs, labels)
    v2 = torch.div(labels, outputs)
    delta = torch.max(v1, v2)

    # Compute the number of occurences smaller than teh treshold
    occurences = (delta < 1.25**p).sum().type(torch.FloatTensor)

    #Compute teh number of items
    n = float(delta.numel())

    return occurences/n


"""
"
" NORMAL
"
"""


def supervised_normal(outputs, labels):
    # Size
    batch_size, channels, height, width = labels.shape

    # Number of element in an image
    n = batch_size * channels * height * width

    # Flat the tensor
    flat_outputs = outputs.view(outputs.numel())
    flat_labels = labels.view(labels.numel())

    # Cross product of the vectors
    dot_product = torch.dot(flat_outputs, flat_labels)

    # Division by number of pixels
    L_normal = dot_product.div(n)

    # Return negative value
    return L_normal