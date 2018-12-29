#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Provides a simple atomic variable:

>>> a = Atomic(0)

Retrieve the value
>>> b = a()
Set a new value:
>>> a <<= b
Get conversion to boolean:
>>> if a:
       do something

Atomic increment:
>>> A += 1

@author: J.A. de Jong - ASCEE
"""
from threading import Lock


class Atomic:
    def __init__(self, val):
        self._val = val
        self._lock = Lock()

    def __iadd__(self, toadd):
        with self._lock:
            self._val += toadd
        return self

    def __isub__(self, toadd):
        with self._lock:
            self._val -= toadd
        return self

    def __bool__(self):
        with self._lock:
            return self._val

    def __ilshift__(self, other):
        with self._lock:
            self._val = other
        return self

    def __call__(self):
        with self._lock:
            return self._val
