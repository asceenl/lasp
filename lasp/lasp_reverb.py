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

    def __init__(self, fs, level, channel=0):
        """
        Initialize Reverberation time computer.

        Args:
            fs: Sampling frequency [Hz]
            level: (Optionally weighted) level values as a function of time, in
            dB.
            channel: Channel index to compute from

        """
        assert level.ndim == 2, 'Invalid number of dimensions in level'

        self._level = level[:, channel][:, np.newaxis]
        # Number of time samples
        self._channel = channel
        self._N = self._level.shape[0]
        self._t = getTime(fs, self._N, 0)
        print(f't: {self._t}')

    def compute(self, istart, istop):
        """
        Compute the reverberation time using a least-squares solver

        Args:
            istart: Start time index reverberation interval
            istop: Stop time index of reverberation interval

        Returns:
            dictionary with result values, contains:
            - istart: start index of reberberation interval
            - istop: stop index of reverb. interval
            - T60: Reverberation time
            - const: Constant value
            - derivative: rate of change of the level in dB/s.
        """


        points = self._level[istart:istop]
        x = self._t[istart:istop][:, np.newaxis]

        # Solve the least-squares problem, by creating a matrix of
        A = np.hstack([x, np.ones(x.shape)])

        print(A.shape)
        print(points.shape)

        # derivative is dB/s of increase/decrease
        sol, residuals, rank, s = np.linalg.lstsq(A, points)


        print(f'sol: {sol}')
        # Derivative of the decay in dB/s
        derivative = sol[0][0]

        # Start level in dB
        const = sol[1][0]

        # Time to reach a decay of 60 dB (reverb. time)
        T60 = -60./derivative
        res = {'istart': istart,
                'istop': istop,
                'const': const,
                'derivative': derivative,
                'T60': T60}
        print(res)
        return res
