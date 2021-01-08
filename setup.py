#!/usr/bin/env python
# coding: utf8

from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))
readme_path = path.join(here, 'README.md')

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open(readme_path, 'r') as f:
    readme = f.read()

setup(
    name='forecaster',
    version='0.1',
    long_description=readme,
    long_description_content_type='text/markdown',
    scripts=['forecaster'],
    license='MIT License',
    packages=[
        'forecaster',
        'forecaster.data',
        'forecaster.engine',
        'forecaster.postprocessors',
        'forecaster.preprocessors',
        'forecaster.visualizer',
    ],
    entry_points={
        'console_scripts': ['forecaster=forecaster.__main__:entrypoint']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=required,
)
