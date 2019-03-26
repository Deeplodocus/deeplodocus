import os
from setuptools import find_packages, setup
from deeplodocus import __version__


# Dynamically calculate the version based on django.VERSION.
#version = __import__('deeplodocus').get_version()
version = __version__

def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()

EXCLUDE_FROM_PACKAGES = ['deeplodocus.bin']

setup(
    name='Deeplodocus',
    version=version,
    python_requires='>=3.5.3',
    url='https://www.deeplodocus.github.io/',
    author='Alix Leroy and Samuel Westlake',
    author_email='deeplodocus@gmail.com',
    description=('The  Deep Learning framework keeping your head above water'),
    long_description=read('README.rst'),
    license='MIT',
    #packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    packages=find_packages(),
    include_package_data=True,
    scripts=['deeplodocus/bin/deeplodocus-admin.py'],
    entry_points={'console_scripts': [
        'deeplodocus = deeplodocus.core.management:execute_from_command_line',
    ]},
    install_requires=['numpy>=1.15.1',
                      'pyyaml>=3.13',
                      'pandas>=0.23.1',
                      'matplotlib>=2.2.2',
                      'aiohttp>=3.4.0',
                      'aiohttp_jinja2>=1.1.0',
                      'psutil>=5-4.8',
                      'graphviz',
                      'pydot'],
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
