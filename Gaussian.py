# -*- coding: utf-8 -*-
""" Base class for an arm class."""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.6"

from random import gauss
from numpy.random import standard_normal
import numpy as np
from scipy.special import erf


class Arm(object):
    """ Base class for an arm class."""

    def __init__(self, lower=0., amplitude=1.):
        """ Base class for an arm class."""
        self.lower = lower  #: Lower value of rewards
        self.amplitude = amplitude  #: Amplitude of value of rewards
        self.min = lower  #: Lower value of rewards
        self.max = lower + amplitude  #: Higher value of rewards

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        if hasattr(self, 'lower') and hasattr(self, 'amplitude'):
            return self.lower, self.amplitude
        elif hasattr(self, 'min') and hasattr(self, 'max'):
            return self.min, self.max - self.min
        else:
            raise NotImplementedError("This method lower_amplitude() has to be implemented in the class inheriting from Arm.")

    # --- Printing

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dir__)

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample."""
        raise NotImplementedError("This method draw(t) has to be implemented in the class inheriting from Arm.")

    def oracle_draw(self, t = None):
        # draw the arm as usual but return the mean
        assert hasattr(self , "mean"), "oracle_draw can be used on Arm with self.mean"
        mean = self.mean
        self.draw(t)
        return mean

    def set_mean_param(self,mean):
        raise NotImplementedError("This method draw(t) has to be implemented in the class inheriting from Arm.")



    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        raise NotImplementedError("This method draw_nparray(t) has to be implemented in the class inheriting from Arm.")

    # --- Lower bound

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        raise NotImplementedError("This method kl(x, y) has to be implemented in the class inheriting from Arm.")

    @staticmethod
    def oneLR(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - mu) / KL(mu, mumax). """
        raise NotImplementedError("This method oneLR(mumax, mu) has to be implemented in the class inheriting from Arm.")

    @staticmethod
    def oneHOI(mumax, mu):
        """ One term for the HOI factor for this arm."""
        return 1 - (mumax - mu)

VARIANCE = 0.05

def klGauss(x, y, sig2x=0.25, sig2y=None):
    r""" Kullback-Leibler divergence for Gaussian distributions of means ``x`` and ``y`` and variances ``sig2x`` and ``sig2y``, :math:`\nu_1 = \mathcal{N}(x, \sigma_x^2)` and :math:`\nu_2 = \mathcal{N}(y, \sigma_x^2)`:
    .. math:: \mathrm{KL}(\nu_1, \nu_2) = \frac{(x - y)^2}{2 \sigma_y^2} + \frac{1}{2}\left( \frac{\sigma_x^2}{\sigma_y^2} - 1 \log\left(\frac{\sigma_x^2}{\sigma_y^2}\right) \right).
    See https://en.wikipedia.org/wiki/Normal_distribution#Other_properties
    - By default, sig2y is assumed to be sig2x (same variance).
    .. warning:: The C version does not support different variances.
    >>> klGauss(3, 3)
    0.0
    >>> klGauss(3, 6)
    18.0
    >>> klGauss(1, 2)
    2.0
    >>> klGauss(2, 1)  # And this KL is symmetric
    2.0
    >>> klGauss(4, 2)
    8.0
    >>> klGauss(6, 8)
    8.0
    - x, y can be negative:
    >>> klGauss(-3, 2)
    50.0
    >>> klGauss(3, -2)
    50.0
    >>> klGauss(-3, -2)
    2.0
    >>> klGauss(3, 2)
    2.0
    - With other values for `sig2x`:
    >>> klGauss(3, 3, sig2x=10)
    0.0
    >>> klGauss(3, 6, sig2x=10)
    0.45
    >>> klGauss(1, 2, sig2x=10)
    0.05
    >>> klGauss(2, 1, sig2x=10)  # And this KL is symmetric
    0.05
    >>> klGauss(4, 2, sig2x=10)
    0.2
    >>> klGauss(6, 8, sig2x=10)
    0.2
    - With different values for `sig2x` and `sig2y`:
    >>> klGauss(0, 0, sig2x=0.25, sig2y=0.5)  # doctest: +ELLIPSIS
    -0.0284...
    >>> klGauss(0, 0, sig2x=0.25, sig2y=1.0)  # doctest: +ELLIPSIS
    0.2243...
    >>> klGauss(0, 0, sig2x=0.5, sig2y=0.25)  # not symmetric here!  # doctest: +ELLIPSIS
    1.1534...
    >>> klGauss(0, 1, sig2x=0.25, sig2y=0.5)  # doctest: +ELLIPSIS
    0.9715...
    >>> klGauss(0, 1, sig2x=0.25, sig2y=1.0)  # doctest: +ELLIPSIS
    0.7243...
    >>> klGauss(0, 1, sig2x=0.5, sig2y=0.25)  # not symmetric here!  # doctest: +ELLIPSIS
    3.1534...
    >>> klGauss(1, 0, sig2x=0.25, sig2y=0.5)  # doctest: +ELLIPSIS
    0.9715...
    >>> klGauss(1, 0, sig2x=0.25, sig2y=1.0)  # doctest: +ELLIPSIS
    0.7243...
    >>> klGauss(1, 0, sig2x=0.5, sig2y=0.25)  # not symmetric here!  # doctest: +ELLIPSIS
    3.1534...
    .. warning:: Using :class:`Policies.klUCB` (and variants) with :func:`klGauss` is equivalent to use :class:`Policies.UCB`, so prefer the simpler version.
    """
    if sig2y is None or - eps < (sig2y - sig2x) < eps:
        return (x - y) ** 2 / (2. * sig2x)
    else:
        return (x - y) ** 2 / (2. * sig2y) + 0.5 * ((sig2x/sig2y)**2 - 1 - log(sig2x/sig2y))

import numpy as np

class Gaussian(Arm):
    """ Gaussian distributed arm, possibly truncated.
    - Default is to truncate into [0, 1] (so Gaussian.draw() is in [0, 1]).
    """

    def __init__(self, mu, sigma=VARIANCE, mini=0, maxi=1):
        """New arm."""
        self.mu = self.mean = mu  #: Mean of Gaussian arm
        assert sigma > 0, "Error, the parameter 'sigma' for Gaussian arm has to be > 0."
        self.sigma = sigma  #: Variance of Gaussian arm
        assert mini <= maxi, "Error, the parameter 'mini' for Gaussian arm has to < 'maxi'."  # DEBUG
        self.min = mini  #: Lower value of rewards
        self.max = maxi  #: Higher value of rewards
        # XXX if needed, compute the true mean : Cf. https://en.wikipedia.org/wiki/Truncated_normal_distribution#Moments
        # real_mean = mu + sigma * (phi(mini) - phi(maxi)) / (Phi(maxi) - Phi(mini))

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return min(max(gauss(self.mu, self.sigma), self.min), self.max)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return np.minimum(np.maximum(self.mu + self.sigma * standard_normal(shape), self.min), self.max)

    def set_mean_param(self, mean):
        self.mu = self.mean = mean

    # --- Printing

    # This decorator @property makes this method an attribute, cf. https://docs.python.org/3/library/functions.html#property
    @property
    def lower_amplitude(self):
        """(lower, amplitude)"""
        return self.min, self.max - self.min

    def __str__(self):
        return "Gaussian"

    def __repr__(self):
        return "N({:.3g}, {:.3g})".format(self.mu, self.sigma)

    # --- Lower bound

    def kl(self, x, y):
        """ The kl(x, y) to use for this arm."""
        return klGauss(x, y, self.sigma)

    def oneLR(self, mumax, mu):
        """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - mu) / KL(mu, mumax). """
        return (mumax - mu) / klGauss(mu, mumax, self.sigma)

    def oneHOI(self, mumax, mu):
        """ One term for the HOI factor for this arm."""
        return 1 - (mumax - mu) / self.max