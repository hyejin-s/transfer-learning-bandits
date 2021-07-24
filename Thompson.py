# -*- coding: utf-8 -*-
""" Base class for any policy.
- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""
from __future__ import division, print_function  # Python 2 compatibility

__author__ = "Lilian Besson"
__version__ = "0.9"

import numpy as np


#: If True, every time a reward is received, a warning message is displayed if it lies outsides of ``[lower, lower + amplitude]``.
CHECKBOUNDS = True
CHECKBOUNDS = False

class BasePolicy(object):
    """ Base class for any policy."""

    def __init__(self, nbArms, lower=0., amplitude=1.):
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

    if CHECKBOUNDS:
        # XXX useless checkBounds feature
        def getReward(self, arm, reward):
            """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
            self.t += 1
            self.pulls[arm] += 1
            # XXX we could check here if the reward is outside the bounds
            if not 0 <= reward - self.lower <= self.amplitude:
                print("Warning: {} received on arm {} a reward = {:.3g} that is outside the interval [{:.3g}, {:.3g}] : the policy will probably fail to work correctly...".format(self, arm, reward, self.lower, self.lower + self.amplitude))  # DEBUG
            # else:
            #     print("Info: {} received on arm {} a reward = {:.3g} that is inside the interval [{:.3g}, {:.3g}]".format(self, arm, reward, self.lower, self.lower + self.amplitude))  # DEBUG
            reward = (reward - self.lower) / self.amplitude
            self.rewards[arm] += reward
    else:
        # It's faster to define two methods and pick one
        # (one test in init, that's it)
        # rather than doing the test in the method
        def getReward(self, arm, reward):
            """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
            self.t += 1
            self.pulls[arm] += 1
            reward = (reward - self.lower) / self.amplitude
            self.rewards[arm] += reward

    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Not defined."""
        raise NotImplementedError("This method choice() has to be implemented in the child class inheriting from BasePolicy.")

    # def handleCollision(self, arm, reward=None):
    #     """ Default to give a 0 reward (or ``self.lower``)."""
    #     # print("DEBUG BasePolicy.handleCollision({}, {}) was called...".format(arm, reward))  # DEBUG
    #     # self.getReward(arm, self.lower if reward is None else reward)
    #     self.getReward(arm, self.lower)
    #     # raise NotImplementedError("This method handleCollision() has to be implemented in the child class inheriting from BasePolicy.")

    # --- Others choice...() methods, partly implemented

    def choiceWithRank(self, rank=1):
        """ Not defined."""
        if rank == 1:
            return self.choice()
        else:
            raise NotImplementedError("This method choiceWithRank(rank) has to be implemented in the child class inheriting from BasePolicy.")

    def choiceFromSubSet(self, availableArms='all'):
        """ Not defined."""
        if availableArms == 'all':
            return self.choice()
        else:
            raise NotImplementedError("This method choiceFromSubSet(availableArms) has to be implemented in the child class inheriting from BasePolicy.")

    def choiceMultiple(self, nb=1):
        """ Not defined."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            raise NotImplementedError("This method choiceMultiple(nb) has to be implemented in the child class inheriting from BasePolicy.")

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        """ Not defined."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            return self.choiceMultiple(nb=nb)

    def estimatedOrder(self):
        """ Return the estimate order of the arms, as a permutation on [0..K-1] that would order the arms by increasing means.
        - For a base policy, it is completely random.
        """
        return np.random.permutation(self.nbArms)

# -*- coding: utf-8 -*-
""" Generic index policy.
- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""

__author__ = "Lilian Besson"
__version__ = "0.6"

import numpy as np

# try:
#     from .BasePolicy import BasePolicy
# except (ImportError, SystemError):
#     from BasePolicy import BasePolicy


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

    # --- Others choice...() methods

    def choiceWithRank(self, rank=1):
        """ In an index policy, choose an arm with index is the (1+rank)-th best (uniformly at random).
        - For instance, if rank is 1, the best arm is chosen (the 1-st best).
        - If rank is 4, the 4-th best arm is chosen.
        .. note:: This method is *required* for the :class:`PoliciesMultiPlayers.rhoRand` policy.
        """
        if rank == 1:
            return self.choice()
        else:
            assert rank >= 1, "Error: for IndexPolicy = {}, in choiceWithRank(rank={}) rank has to be >= 1.".format(self, rank)
            self.computeAllIndex()
            sortedRewards = np.sort(self.index)
            # Question: What happens here if two arms has the same index, being the max?
            # Then it is fair to chose a random arm with best index, instead of aiming at an arm with index being ranked rank
            chosenIndex = sortedRewards[-rank]
            # Uniform choice among the rank-th best arms
            try:
                return np.random.choice(np.nonzero(self.index == chosenIndex)[0])
            except ValueError:
                print("Warning: unknown error in IndexPolicy.choiceWithRank(): the indexes were {} but couldn't be used to select an arm.".format(self.index))
                return np.random.randint(self.nbArms)

    def choiceFromSubSet(self, availableArms='all'):
        """ In an index policy, choose the best arm from sub-set availableArms (uniformly at random)."""
        if isinstance(availableArms, str) and availableArms == 'all':
            return self.choice()
        # If availableArms are all arms? XXX no this could loop, better do it here
        # elif len(availableArms) == self.nbArms:
        #     return self.choice()
        elif len(availableArms) == 0:
            print("WARNING: IndexPolicy.choiceFromSubSet({}): the argument availableArms of type {} should not be empty.".format(availableArms, type(availableArms)))  # DEBUG
            # WARNING if no arms are tagged as available, what to do ? choose an arm at random, or call choice() as if available == 'all'
            return self.choice()
        else:
            for arm in availableArms:
                self.index[arm] = self.computeIndex(arm)
            # Uniform choice among the best arms
            try:
                return availableArms[np.random.choice(np.nonzero(self.index[availableArms] == np.max(self.index[availableArms]))[0])]
            except ValueError:
                return np.random.choice(availableArms)

    def choiceMultiple(self, nb=1):
        """ In an index policy, choose nb arms with maximal indexes (uniformly at random)."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            self.computeAllIndex()
            sortedIndexes = np.sort(self.index)
            # Uniform choice of nb different arms among the best arms
            # FIXED sort it then apply affectation_order, to fix its order ==> will have a fixed nb of switches for CentralizedMultiplePlay
            try:
                return np.random.choice(np.nonzero(self.index >= sortedIndexes[-nb])[0], size=nb, replace=False)
            except ValueError:
                return np.random.choice(self.nbArms, size=nb, replace=False)

    def choiceIMP(self, nb=1, startWithChoiceMultiple=True):
        """ In an index policy, the IMP strategy is hybrid: choose nb-1 arms with maximal empirical averages, then 1 arm with maximal index. Cf. algorithm IMP-TS [Komiyama, Honda, Nakagawa, 2016, arXiv 1506.00779]."""
        if nb == 1:
            return np.array([self.choice()])
        else:
            # For first exploration steps, do pure exploration
            if startWithChoiceMultiple:
                if np.min(self.pulls) < 1:
                    return self.choiceMultiple(nb=nb)
                else:
                    empiricalMeans = self.rewards / self.pulls
            else:
                empiricalMeans = self.rewards / self.pulls
                empiricalMeans[self.pulls < 1] = float('inf')
            # First choose nb-1 arms, from rewards
            sortedEmpiricalMeans = np.sort(empiricalMeans)
            exploitations = np.random.choice(np.nonzero(empiricalMeans >= sortedEmpiricalMeans[-nb])[0], size=nb - 1, replace=False)
            # Then choose 1 arm, from index now
            availableArms = np.setdiff1d(np.arange(self.nbArms), exploitations)
            exploration = self.choiceFromSubSet(availableArms)
            # Affect a random location to is exploratory arm
            return np.insert(exploitations, np.random.randint(np.size(exploitations) + 1), exploration)

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

# -*- coding: utf-8 -*-
""" Basic Bayesian index policy. By default, it uses a Beta posterior. """

__author__ = "Lilian Besson"
__version__ = "0.9"


from SMPyBandits.Policies.Posterior import Beta

# try:
#     from .IndexPolicy import IndexPolicy
#     from .Posterior import Beta
# except ImportError:
#     from IndexPolicy import IndexPolicy
#     from Posterior import Beta


class BayesianIndexPolicy(IndexPolicy):
    """ Basic Bayesian index policy.
    - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
    - Use ``*args`` and ``**kwargs`` if you want to give parameters to the underlying posteriors.
    - Or use ``params_for_each_posterior`` as a *list* of parameters (as a dictionary) to give a different set of parameters for each posterior.
    """

    def __init__(self, nbArms,
            posterior=Beta,
            lower=0., amplitude=1.,
            *args, **kwargs
        ):
        """ Create a new Bayesian policy, by creating a default posterior on each arm."""
        super(BayesianIndexPolicy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.posterior = [None] * nbArms  #: Posterior for each arm. List instead of dict, quicker access
        if 'params_for_each_posterior' in kwargs:
            params = kwargs['params_for_each_posterior']
            print("'params_for_each_posterior' is in kwargs, so using params =\n{}\nas a list of parameters to give to each posterior.".format(params))  # DEBUG
            for arm in range(self.nbArms):
                print("Creating posterior for arm {}, with params = {}.".format(arm, params[arm]))  # DEBUG
                self.posterior[arm] = posterior(**params[arm])
        else:
            for arm in range(self.nbArms):
                # print("Creating posterior for arm {}, with args = {} and kwargs = {}.".format(arm, args, kwargs))  # DEBUG
                self.posterior[arm] = posterior(*args, **kwargs)
        self._posterior_name = str(self.posterior[0].__class__.__name__)
        self._kwargs = kwargs
        self.LC_value = 0
        self.eta_solution = 0
        self.eta_compare = 0
        self.zeta_info = 0
        self.compare_info = 0

    def __str__(self):
        """ -> str"""
        if self._posterior_name == "Beta":
            return "{}".format(self.__class__.__name__)
        else:
            return "{}({})".format(self.__class__.__name__, self._posterior_name)

    def startGame(self):
        """ Reset the posterior on each arm."""
        self.t = 0
        for arm in range(self.nbArms):
            self.posterior[arm].reset()
        # print("Policy {} reinitialized with posteriors: {}".format(self, [str(p) for p in self.posterior])) # DEBUG

    def getReward(self, arm, reward):
        """ Update the posterior on each arm, with the normalized reward."""
        self.posterior[arm].update((reward - self.lower) / self.amplitude)
        self.t += 1

    def computeIndex(self, arm):
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from BayesianIndexPolicy.")




# -*- coding: utf-8 -*-
""" The Thompson (Bayesian) index policy.
- By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
- Reference: [Thompson - Biometrika, 1933].
"""

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann, Lilian Besson"
__version__ = "0.9"

# try:
#     from .BayesianIndexPolicy import BayesianIndexPolicy
# except (ImportError, SystemError):
#     from BayesianIndexPolicy import BayesianIndexPolicy


class Thompson(BayesianIndexPolicy):
    r"""The Thompson (Bayesian) index policy.
    - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
    - Prior is initially flat, i.e., :math:`a=\alpha_0=1` and :math:`b=\beta_0=1`.
    - A non-flat prior for each arm can be given with parameters ``a`` and ``b``, for instance::
        nbArms = 2
        prior_failures  = a = 100
        prior_successes = b = 50
        policy = Thompson(nbArms, a=a, b=b)
        np.mean([policy.choice() for _ in range(1000)])  # 0.515 ~= 0.5: each arm has same prior!
    - A different prior for each arm can be given with parameters ``params_for_each_posterior``, for instance::
        nbArms = 2
        params0 = { 'a': 10, 'b': 5}  # mean 1/3
        params1 = { 'a': 5, 'b': 10}  # mean 2/3
        params = [params0, params1]
        policy = Thompson(nbArms, params_for_each_posterior=params)
        np.mean([policy.choice() for _ in range(1000)])  # 0.9719 ~= 1: arm 1 is better than arm 0 !
    - Reference: [Thompson - Biometrika, 1933].
    """

    def __str__(self):
        return "Thompson Sampling"
    
    def computeIndex(self, arm):
        r""" Compute the current index, at time t and after :math:`N_k(t)` pulls of arm k, giving :math:`S_k(t)` rewards of 1, by sampling from the Beta posterior:
        .. math::
            A(t) &\sim U(\arg\max_{1 \leq k \leq K} I_k(t)),\\
            I_k(t) &\sim \mathrm{Beta}(1 + \tilde{S_k}(t), 1 + \tilde{N_k}(t) - \tilde{S_k}(t)).
        """
        return self.posterior[arm].sample()