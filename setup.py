# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

requires = [
        'nose',
        'sphinx',
        'ujson',
        ]

setup(
    name='challenge',
    version='0.0.1',
    description='',
    long_description=readme,
    author='Greeshma Swaminathan <gswamina@ucsc.edu>, Neha Ojha <nojha@ucsc.edu>, Jianshen Liu <jliu120@ucsc.edu>, Alex Bardales <abardale@ucsc.edu>',
    url='https://github.com/ljishen/yelp-dataset-challenge',
    license=license,
    packages=find_packages(exclude=('docs')),
    install_requires=requires,
    tests_require=requires
)
