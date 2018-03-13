#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:24:52 2018

@author: zhong
"""

from setuptools import setup, find_packages

setup(name='nuclei',
      version='0.1',
      description='The package for the nuclei kaggle competition',
#      url='http://github.com/storborg/funniest',
      author='Zhichao Zhong',
      author_email='zhchzhong@gmail.com',
      license='MIT',
#      packages=['nuclei'],
      package=find_packages(),
      zip_safe=False)