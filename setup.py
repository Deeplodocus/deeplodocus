import os
import sys
from distutils.sysconfig import get_python_lib

from setuptools import find_packages, setup


CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 5)

# This check and everything above must remain compatible with Python 2.7.
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
    ==========================
    Unsupported Python version
    ==========================
    This version of Deeplodocus requires Python {}.{}, but you're trying to
    install it on Python {}.{}.
    This may be because you are using a version of pip that doesn't
    understand the python_requires classifier. Make sure you
    have pip >= 9.0 and setuptools >= 24.2, then try again:
        $ python -m pip install --upgrade pip setuptools
        $ python -m pip install deeplodocus
    This will install the latest version of Django which works on your
    version of Python. 
    """.format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))

    sys.exit(1)



# Warn if we are installing over top of an existing installation. This can
# cause issues where files that were deleted from a more recent Deeplodocus are
# still present in site-packages.

old_version_installed = False

#
if "install" in sys.argv:

    lib_paths = [get_python_lib()]

    if lib_paths[0].startswith("/usr/lib/"):
        # We have to try also with an explicit prefix of /usr/local in order to
        # catch Debian's custom user site-packages directory.
        lib_paths.append(get_python_lib(prefix="/usr/local"))

    for lib_path in lib_paths:
        existing_path = os.path.abspath(os.path.join(lib_path, "deeplodocus"))

        if os.path.exists(existing_path):
            # We note the need for the warning here, but present it after the
            # command is run, so it's more likely to be seen.
            old_version_installed = True



if old_version_installed:
    sys.stderr.write("""
    ========
    WARNING!
    ========
    You have just installed Django over top of an existing
    installation, without removing it first. Because of this,
    your install may now include extraneous files from a
    previous version that have since been removed from
    Django. This is known to cause a variety of problems. You
    should manually remove the
    %(existing_path)s
    directory and re-install Django.
    """ % {"existing_path": existing_path})

# Dynamically calculate the version based on django.VERSION.
version = __import__('django').get_version()


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
    install_requires=['pytorch >= 0.4.0'],
    extras_require={
        "cv2": ["cv2 >= 3.4.1"],
        "numpy": ["nmpy >= 1.14.3"],
    },
    zip_safe=False,
    classifiers=[
        'Development Status :: 0.1.0 - Pre-Alpha',
        'Environment ::  Environment',
        'Framework :: Deeplodocus',
        'Intended Audience :: Deep Learning Researchers / Engineers',
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
        'Documentation': 'https://docs.djangoproject.com/',
        'Funding': 'https://www.djangoproject.com/fundraising/',
        'Source': 'https://github.com/django/django',
        'Tracker': 'https://code.djangoproject.com/',
    },
)
