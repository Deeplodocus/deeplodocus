# What is Deeplodocus?

In one line, Deeplodocus is a modular, flexible and accessible environment for the structured and rapid development of deep learning projects. 

The challenge of training, testing and deploying artificial neural networks involves the completion and integration of numerous complex sub-routines. Efficient data loading and augmentation, model design and optimisation procedures are all non-trivial tasks that require structured management throughout any deep learning project. However, approaching each one of these tasks from scratch is not always necessary—especially given the curent range of machine learning libraries—and instead the development of only select modules is often be more efficient. Therefore, Deeplodocus is a flexible environment that enables users to seamlessly integrate any number of existing and custom modules to rapidly build powerful deep learning projects.

Not just another abstract API, Deeplodocus is a new approach to structuring deep learning projects. It allows the level of abstraction in each domain of a project to be tailored to suit the user requirements. Users can quickly compile new deep learning pipelines solely from pre-implemented modules without writing a single line of code, or investigate novel cutting-edge techiques by taking control of every low-level operation. 

Built on PyTorch, Deeplodocus offers comprehensive control of high-level configurations and parameters, whilst maintaining maximum flexibility through modularity to accelerate the rapid-prototyping of deep learning techniques. 

## Installation

Deeplodocus is compatible with versions of Python 3.6 and onward, and can be installed with PIP. However, before installing Deeplodocus, we recommend that you install PyTorch. 

### Install PyTorch

Deeplodocus is built around PyTorch, however Pytorch does not come packaged with Deeplodocus as its installation depends on your version of [CUDA](https://developer.nvidia.com/cuda-downloads). To install PyTorch, we recommend that you follow instructions from the official [PyTorch website](https://pytorch.org/). 

### Install Deeplodocus from PyPI

For the latest stable release of Deeplodocus, we recommend installing from PyPI. Simply enter the command below into your terminal:

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
	
1. Install this local repository, either:

	With pip:

	```
	$ pip install <path-to-deeplodocus>
	```
	
	Or, with pip in development mode (so Deeplodocus appears to be installed but is still editable from the local source package), with:

	```
	$ pip install -e <path-to-deeplodocus>
	```
	
## Support 

Users are welcome to post bug reports and feature requests in [GitHub issues](https://github.com/Deeplodocus/deeplodocus/issues).