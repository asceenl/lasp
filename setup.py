#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: J.A. de Jong - ASCEE
"""
from setuptools import setup, find_packages


# class CMakeExtension(Extension):
#     """
#     An extension to run the cmake build
#
#     This simply overrides the base extension class so that setuptools
#     doesn't try to build your sources for you
#     """
#
#     def __init__(self, name, sources=[]):
#
#         super().__init__(name=name, sources=sources)


setup(
      name="LASP",
      version="1.0",
      packages=find_packages(),
      long_description=open("./README.md", 'r').read(),
      long_description_content_type="text/markdown",
      # ext_modules=[CMakeExtension('lasp/wrappers.so'),
      #              ],
      package_data={'lasp': ['wrappers.so']},
      author='J.A. de Jong - ASCEE',
      author_email="j.a.dejong@ascee.nl",
      install_requires=['matplotlib>=1.0',
                        'scipy>=1.0', 'numpy>=1.0', 'h5py',
                        ],
      license='MIT',
      description="Library for Acoustic Signal Processing",
      keywords="",
      url="https://www.ascee.nl/lasp/",   # project home page, if any

)
