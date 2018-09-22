import os
from setuptools import find_packages, setup


REQUIRED_PYTHON = (3, 5)



# Dynamically calculate the version based on django.VERSION.
version = __import__('deeplodocus').get_version()


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

EXCLUDE_FROM_PACKAGES = ['deeplodocus.bin']


setup(
    name='Deeplodocus',
    version=version,
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    url='https://www.deeplodocus.github.io/',
    author='Alix Leroy and Samuel Westlake',
    author_email='deeplodocus@gmail.com',
    description=('Deeplodocus is a high-level Python framework for Deep Learning that encourages rapid neural networks  trainings'),
    long_description=read('README.rst'),
    license='MIT',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    scripts=['deeplodocus/bin/deeplodocus.py'],
    entry_points={'console_scripts': [
        'deeplodocus = deeplodocus.core.management:execute_from_command_line',
    ]},
    install_requires=['torch>=0.4.0',
                      'numpy>=1.9.1',
                      'pyyaml',
                      'pandas>=0.23.0',
                      'matplotlib>=2.2.2',
                      'aiohttp>=3.4.1'],
    extras_require={
        "cv2": ["opencv-python >= 3.4.1"]
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development :: Libraries :: Application Frameworks',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    project_urls={
        'Documentation': 'https://www.deeplodocus.github.io/',
        'Source': 'https://github.com/Deeplodocus/deeplodocus',
    },
)
