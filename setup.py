#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J.A. de Jong - ASCEE
"""
from setuptools import setup

descr = "Python wrappers around several C++ optimized beamforming codes"


setup(
      name="LASP",
      version="1.0",
      packages=['lasp'],
      author='J.A. de Jong - ASCEE',
      author_email="j.a.dejong@ascee.nl",
      # Project uses reStructuredText, so ensure that the docutils get
      # installed or upgraded on the target machine
      install_requires=['matplotlib>=1.0', 'scipy>=1.0', 'numpy>=1.0'],
      license='MIT',
      description=descr,
      keywords="",
      url="http://www.ascee.nl/lasp/",   # project home page, if any

)
