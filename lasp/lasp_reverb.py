# -*- coding: utf-8 -*-
"""!
Author: J.A. de Jong - ASCEE

Description:
Reverberation time estimation tool using least squares
"""
from .lasp_common import getTime
import numpy as np


class ReverbTime:
    """
    Tool to estimate the reverberation time
    """

    def __init__(self, fs, level):
        """
        Initialize Reverberation time computer.

        Args:
            fs: Sampling frequency [Hz]
            level: (Optionally weighted) level values as a function of time, in
            dB.

        """
        assert level.ndim == 1, 'Invalid number of dimensions in level'
        self._level = level
        # Number of time samples
        self._N = self._level.shape[0]
        self._t = getTime(fs, self._N, 0)

    def compute(self, tstart, tstop):
        """
        Compute the reverberation time using a least-squares solver

        Args:
            tstart: Start time of reverberation interval
            stop: Stop time of reverberation interval

        Returns:
            dictionary with result values, contains:
            - istart: start index of reberberation interval
            - istop: stop index of reverb. interval
            - T60: Reverberation time
            - const: Constant value
            - derivative: rate of change of the level in dB/s.
        """

        # Find start and stop indices. Coerce if they are outside
        # of the valid domain
        istart = np.where(self._t >= tstart)[0][0]
        istop = np.where(self._t >= tstop)[0]

        if istop.shape[0] == 0:
            istop = self._level.shape[0]
        else:
            istop = istop[0]

        points = self._level[istart:istop]
        x = self._t[istart:istop][:, np.newaxis]

        # Solve the least-squares problem, by creating a matrix of
        A = np.hstack([x, np.ones((x.shape))])

        # derivative is dB/s of increase/decrease
        sol, residuals, rank, s = np.linalg.lstsq(A, points)

        # Derivative of the decay in dB/s
        derivative = sol[0]

        # Start level in dB
        const = sol[1]

        # Time to reach a decay of 60 dB (reverb. time)
        T60 = -60./derivative

        return {'istart': istart,
                'istop': istop,
                'const': const,
                'derivative': derivative,
                'T60': T60}
