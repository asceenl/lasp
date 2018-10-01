#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description:
"""
__all__ = ['report_quality', 'getReportQuality']

_report_quality = False


def getReportQuality():
    global _report_quality
    return _report_quality


def report_quality():
    import matplotlib as mpl
    # mpl.use('Qt5Agg')
    global _report_quality
    # mpl.use("pgf")
    # rc('font',**{'family':'serif','serif':['Libertine']})
    # for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    # rc('text', usetex=True)
    # TeX preamble
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
    mpl.rcParams.update(params)

    _report_quality = True
