# Home

## About Deeplodocus

Deeplodocus is a modular, flexible and accessible environment for accelerating the start-up and development of deep learning projects. 

Many challenges must be overcome along the course of training, testing and deploying artificial neural networks. Efficient data loading and augmentation, model design and optimisation procedures are all non-trivial tasks that require structured management throughout any deep learning project. However, approaching each one of these tasks from scratch is not always necessary—especially given today's selection of machine learning libraries—and instead the development of only select modules is often be more efficient.

Deeplodocus represents a new approach to prototyping and structuring deep learning projects. Built on PyTorch, Deeplodocus offers comprehensive control of high-level configurations and parameters, whilst maintaining maximum flexibility through modularity.

## Installation

Deeplodocus is compatible with versions of Python 3.6 and onward, and can be installed with PIP. However, before installing Deeplodocus, we recommend that you install PyTorch. 

### Install PyTorch

Deeplodocus is built around PyTorch, however Pytorch does not come packaged with Deeplodocus as its installation depends on your version of [CUDA](https://developer.nvidia.com/cuda-downloads). 

To install PyTorch, we recommend that you follow instructions from the official [PyTorch website](https://pytorch.org/). 

### Install Deeplodocus from PyPI

For the latest stable release of Deeplodocus, we recommend installing from PyPI.

Simply enter the command below into your terminal:

```bash
$ pip3 install deeplodocus
```

### Install Deeplodocus from GitHub

To install the most recent version of Deeplodocus, download the Deeplodocus repository and install from this local source. Note, this version may be less stable than the current PyPI release. 

1. Download the Deeplodocus repository, by either:

	Cloning from GitHub using git with the command below:

	```
	$ git clone https://github.com/Deeplodocus/deeplodocus.git
	``` 
	
	Or by:
	 	
	- Visiting [Deeplodocus on GitHub](https://github.com/Deeplodocus/deeplodocus),
	- Clicking `clone or download` then `Download ZIP`,
	- Unpacking the archive to a desired location.
	
1. Install this local repository, by either:

	Installing normally with:

	```
	$ pip install <path-to-deeplodocus>
	```
	
	Or by:

	Installing in Development Mode (so Deeplodocus appears to be installed but is still editable from the local source package), with:

	```
	$ pip install -e <path-to-deeplodocus>
	```