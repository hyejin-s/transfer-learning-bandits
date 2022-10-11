  
# -*- coding: utf-8 -*-
""" The UCB policy for bounded bandits.
- Reference: [Lai & Robbins, 1985].
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.1"

from math import sqrt, log
import numpy as np
np.seterr(divide='ignore')  # XXX dangerous in general, controlled here!

class BasePolicy(object):
    """ Base class for any policy."""

    def __init__(self, nbArms, lower=0., amplitude=1., **kwargs):
        """ New policy."""
        # Parameters
        assert nbArms > 0, "Error: the 'nbArms' parameter of a {} object cannot be <= 0.".format(self)  # DEBUG
        self.nbArms = nbArms  #: Number of arms
        self.lower = lower  #: Lower values for rewards
        assert amplitude > 0, "Error: the 'amplitude' parameter of a {} object cannot be <= 0.".format(self)  # DEBUG
        self.amplitude = amplitude  #: Larger values for rewards
        # Internal memory
        self.t = 0  #: Internal time
        self.pulls = np.zeros(nbArms, dtype=int)  #: Number of pulls of each arms
        self.rewards = np.zeros(nbArms)  #: Cumulated rewards of each arms

    def __str__(self):
        """ -> str"""
        return self.__class__.__name__

    # --- Start game, and receive rewards

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        self.t = 0
        self.pulls.fill(0)
        self.rewards.fill(0)
        
    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
        self.t += 1
        self.pulls[arm] += 1
        reward = (reward - self.lower) / self.amplitude
        self.rewards[arm] += reward
        
    def choice(self):
        """ Not defined."""
        raise NotImplementedError("This method choice() has to be implemented in the child class inheriting from BasePolicy.")


class IndexPolicy(BasePolicy):
    """ Class that implements a generic index policy."""

    def __init__(self, nbArms, lower=0., amplitude=1.):
        """ New generic index policy.
        - nbArms: the number of arms,
        - lower, amplitude: lower value and known amplitude of the rewards.
        """
        super(IndexPolicy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.index = np.zeros(nbArms)  #: Numerical index for each arms

    # --- Start game, and receive rewards

    def startGame(self):
        """ Initialize the policy for a new game."""
        super(IndexPolicy, self).startGame()
        self.index.fill(0)

    def computeIndex(self, arm):
        """ Compute the current index of arm 'arm'."""
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from IndexPolicy.")

    def computeAllIndex(self):
        """ Compute the current indexes for all arms. Possibly vectorized, by default it can *not* be vectorized automatically."""
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)

    # --- Basic choice() method

    def choice(self):
        r""" In an index policy, choose an arm with maximal index (uniformly at random):
        .. math:: A(t) \sim U(\arg\max_{1 \leq k \leq K} I_k(t)).
        .. warning:: In almost all cases, there is a unique arm with maximal index, so we loose a lot of time with this generic code, but I couldn't find a way to be more efficient without loosing generality.
        """
        # I prefer to let this be another method, so child of IndexPolicy only needs to implement it (if they want, or just computeIndex)
        self.computeAllIndex()
        # Uniform choice among the best arms
        try:
            return np.random.choice(np.nonzero(self.index == np.max(self.index))[0])
        except ValueError:
            print("Warning: unknown error in IndexPolicy.choice(): the indexes were {} but couldn't be used to select an arm.".format(self.index))
            return np.random.randint(self.nbArms)
    
    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means."""
        self.computeAllIndex()
        return np.argsort(self.index)

    def estimatedBestArms(self, M=1):
        """ Return a (non-necessarily sorted) list of the indexes of the M-best arms. Identify the set M-best."""
        assert 1 <= M <= self.nbArms, "Error: the parameter 'M' has to be between 1 and K = {}, but it was {} ...".format(self.nbArms, M)  # DEBUG
        # # WARNING this slows down everything, but maybe the only way to make this correct?
        # if np.all(np.isinf(self.index)):
        #     # Initial guess: random estimate of the set Mbest
        #     choice = np.random.choice(self.nbArms, size=M, replace=False)
        #     print("Warning: estimatedBestArms() for self = {} was called with M = {} but all indexes are +inf, so using a random estimate = {} of Mbest instead of the biased [K-M,...,K-1] ...".format(self, M, choice))  # DEBUG
        #     return choice
        # else:
        order = self.estimatedOrder()
        return order[-M:]


class UCB(IndexPolicy):
    """ The UCB policy for bounded bandits.
    - Reference: [Lai & Robbins, 1985].
    """

    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k:
        .. math:: I_k(t) = \frac{X_k(t)}{N_k(t)} + \sqrt{\frac{2 \log(t)}{N_k(t)}}.
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt((2 * log(self.t)) / self.pulls[arm])

    def computeAllIndex(self):
        """ Compute the current indexes for all arms, in a vectorized manner."""
        indexes = (self.rewards / self.pulls) + np.sqrt((2 * np.log(self.t)) / self.pulls)
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes


# --- Debugging

if __name__ == "__main__":
    # Code for debugging purposes.
    from doctest import testmod
    print("\nTesting automatically all the docstring written in each functions of this module :")
    testmod(verbose=True)

