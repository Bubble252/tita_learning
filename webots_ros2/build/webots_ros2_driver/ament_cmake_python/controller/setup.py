from setuptools import find_packages
from setuptools import setup

setup(
    name='controller',
    version='2023.1.3',
    packages=find_packages(
        include=('controller', 'controller.*')),
)
