#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J.A. de Jong - ASCEE
"""
from setuptools import setup

descr = """Library for Acoustic Signal Processing. This Python module contains
tools and code for common operations on acoustic signals."""

setup(
      name="LASP",
      version="1.0",
      packages=['lasp'],
      author='J.A. de Jong - ASCEE',
      author_email="j.a.dejong@ascee.nl",
      install_requires=['matplotlib>=1.0', 'scipy>=1.0', 'numpy>=1.0'],
      license='MIT',
      description=descr,
      keywords="",
      url="https://www.ascee.nl/lasp/",   # project home page, if any

)
