#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description:
"""
__all__ = ['init_backend', 'getReportQuality']

_report_quality = False
_init = False


def init_matplotlib(report_quality=False):
    global _init
    if not _init:
        _init = True
        print('Initializing matplotlib...')
        preamble = [
              r'\usepackage{libertine-type1}'
              r'\usepackage[libertine]{newtxmath}'
              # r'\usepackage{fontspec}',
              # r'\setmainfont{Libertine}',
        ]
        params = {
            'font.family': 'serif',
            'text.usetex': True,
            'text.latex.unicode': True,
            'pgf.rcfonts': False,
            'pgf.texsystem': 'pdflatex',
            'pgf.preamble': preamble,
        }
        import matplotlib
        matplotlib.rcParams.update(params)
        global _report_quality
        _report_quality = report_quality


def init_backend(report_quality=False):
    global _init
    if not _init:
        import matplotlib
        matplotlib.use('Qt5Agg', warn=False, force=True)
        init_matplotlib(report_quality)
        _init = True
        import matplotlib.pyplot as plt
        plt.ion()


def getReportQuality():
    global _report_quality
    return _report_quality
