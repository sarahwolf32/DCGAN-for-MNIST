from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=1.7']

setup(
    name='DCGAN',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    author='Sarah Wolf',
    description='Generating MNIST characters with a DCGAN'
)