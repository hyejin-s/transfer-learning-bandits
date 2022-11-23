# -*- coding: utf-8 -*-

from enum import Enum  # For the different phases
import numpy as np
from scipy.optimize import linprog
from math import log, sqrt

eps = 1e-15  # Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]
#: Default value for the :math:`\gamma` parameter, 0.0 is a safe default.
GAMMA = 0.001
EPSILON = 1

LCvalue, trueLC = -1, -1    # to print LC

# Bernoulli KL-divergence
def klBern(x, y):
   x = min(max(x, eps), 1 - eps)
   y = min(max(y, eps), 1 - eps)
   return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))

def log_plus(x):
    return max(1, log(x))

#: Different phases during the OSSB algorithm
Phase = Enum('Phase', ['initialisation', 'exploitation', 'estimation', 'exploration'])

def estimate_Lipschitz_constant(thetas, embeddings):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i+1]-thetas[i])/(np.linalg.norm(embeddings[i+1]-embeddings[i])))

    return np.amax(L_values)

def get_confusing_bandit(k, L, thetas, embeddings):
    theta_max = np.amax(thetas)
    # values : \lambda_i^k (arm k,i \in K^{-})
    lambda_values = np.zeros_like(thetas)
    for i, theta in enumerate(thetas):
        lambda_values[i] = max(theta, theta_max-L*abs(np.linalg.norm(embeddings[k]-embeddings[i])))
        if L == np.inf:
            if k == i:
                lambda_values[i] = theta_max
    return lambda_values

def solve_optimization_problem__Lipschitz(thetas, embeddings, L=-1):
    theta_max = np.amax(thetas)
    c = theta_max - thetas  # c : (\theta^*-theta_k)_{k\in K}
    
    sub_arms = (np.nonzero(c))[0]
    opt_arms = (np.where(c==0))[0]

    global LCvalue
    if sub_arms.size == 0:    # ex) arms' mean => all 0
        LCvalue = 0
        return np.full(thetas.size, np.inf)
    # for unknown Lipschitz Constant
    if L == -1:
        LCvalue = estimate_Lipschitz_constant(thetas)
    A_ub = np.zeros((sub_arms.size, sub_arms.size))
    for j, k in enumerate(sub_arms):
        nu = get_confusing_bandit(k, L, thetas, embeddings) # get /lambda^k
        for i, idx in enumerate(sub_arms):         # A_eq[j]=
            A_ub[j][i] = klBern(thetas[idx], nu[idx])
    A_ub = (-1)*A_ub
    b_ub = (-1)*np.ones_like(np.arange(sub_arms.size, dtype=int))
    delta = c[c!=0]
    bounds_sub = np.zeros((sub_arms.size, 2))

    for idx, i in enumerate(np.where(thetas != max(thetas))[0]):
        bounds_sub[idx] = (0, None)

    ## revised simplex
    try:
        res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=bounds_sub)
    except Exception as e:
        print(str(e))
        res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='revised simplex', bounds=bounds_sub)
        if res.success == True: 
            print("LinearProgramming Error_Exception: success")
        else:
            print("LinearProgramming Error_Exception: fail") 
            return np.full(thetas.size, -1)

    if res.success == True: # return res.x
        result = np.zeros(thetas.size)
        for i, idx in enumerate(opt_arms):
            result[idx] = np.inf
        for i, idx in enumerate(sub_arms):
            result[idx] = res.x[i]
        return result
    else: # Fail
        if res.status == 2: # we can ignore this failure
            result = np.zeros(thetas.size)
            for i, idx in enumerate(opt_arms):
                result[idx] = np.inf
            for i, idx in enumerate(sub_arms):
                result[idx] = res.x[i]
            return result
        elif res.status == 4: # numerical difficult error
            # option_again = {'tol':1e-8, 'sym_pos':False, 'cholesky':False, 'lstsq':True}
            print("LinearProgramming Error: Numerical difficulties error")
            res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=bounds_sub)
            # res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='revised simplex', options=option_again)
            if res.success == True: 
                print("LinearProgramming Error4: success")
            else: 
                print("LinearProgramming Error4: fail")
                return np.full(thetas.size, -1)
            
            result = np.zeros(thetas.size)
            for i, idx in enumerate(opt_arms):
                result[idx] = np.inf
            for i, idx in enumerate(sub_arms):
                result[idx] = res.x[i]
            return result
        else:
            print("LinearProgramming Error: Last fail")
            return np.full(thetas.size, -1)

def solve_optimization_problem__classic(thetas, embeddings):
    """ 
    Solve the optimization problem (2)-(3) as defined in the paper, for classical stochastic bandits.

    - No need to solve anything, as they give the solution for classical bandits.
    """
    values = np.zeros_like(thetas)
    theta_max = np.max(thetas)

    for i, theta in enumerate(thetas):
        if theta < theta_max:
            values[i] = 1 / klBern(theta, theta_max)
        else:
            values[i] = np.inf
    return values

##########################################################################################################################
        
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
        self.t = 1  #: Internal time
        self.pulls = np.zeros(nbArms, dtype=int)  #: Number of pulls of each arms
        self.rewards = np.zeros(nbArms)  #: Cumulated rewards of each arms
        self.old_mt = np.zeros(nbArms)
        self.LC_value = 0

    def __str__(self):
        """ -> str"""
        return self.__class__.__name__

    # --- Start game, and receive rewards

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        self.t = 1
        self.pulls.fill(0)
        self.rewards.fill(0)
        self.old_mt = np.full(self.nbArms, 0)
        
    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
        self.t += 1
        self.pulls[arm] += 1
        reward = (reward - self.lower) / self.amplitude
        self.rewards[arm] += reward
        
    def choice(self):
        """ Not defined."""
        raise NotImplementedError("This method choice() has to be implemented in the child class inheriting from BasePolicy.")

# Algorithm2
class OSSB_DEL(BasePolicy):

    def __init__(self, nbArms, embeddings, epsilon=EPSILON, gamma=GAMMA,
                 solve_optimization_problem="classic", LC_value=False,
                 lower=0., amplitude=1., **kwargs):
        super(OSSB_DEL, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # Arguments
        self.embeddings = embeddings
        assert 0 <= epsilon <= 1, "Error: the 'epsilon' parameter for 'OSSB' class has to be 0 <= . <= 1 but was {:.3g}.".format(epsilon)  # DEBUG
        self.epsilon = epsilon  #: Parameter :math:`\varepsilon` for the OSSB algorithm. Can be = 0.
        assert gamma >= 0, "Error: the 'gamma' parameter for 'OSSB' class has to be >= 0. but was {:.3g}.".format(gamma)  # DEBUG
        self.gamma = gamma  #: Parameter :math:`\gamma` for the OSSB algorithm. Can be = 0.
        # Solver for the optimization problem.
        self._solve_optimization_problem = solve_optimization_problem__classic  # Keep the function to use to solve the optimization problem
        self._info_on_solver = ", Bern"  # small delta string

        if solve_optimization_problem == "Lipschitz" and LC_value=="estimated":
            self._info_on_solver = ", Lipschitz, estimated"
            self._solve_optimization_problem = solve_optimization_problem__Lipschitz
        if solve_optimization_problem == "Lipschitz" and LC_value=="true":
            self._info_on_solver = ", Lipschitz, true"
            self._solve_optimization_problem = solve_optimization_problem__Lipschitz
        self._kwargs = kwargs  # Keep in memory the other arguments, to give to self._solve_optimization_problem

    def __str__(self):
        """ -> str"""
        return r"OSSB_gamma($\varepsilon={:.3g}$, $\gamma={:.3g}${})".format(self.epsilon, self.gamma, self._info_on_solver)

    # --- Start game, and receive rewards

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        super(OSSB_DEL, self).startGame()
        self.counter_s_no_exploitation_phase = 0
        self.phase = Phase.initialisation

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
        super(OSSB_DEL, self).getReward(arm, reward)

    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Applies the OSSB procedure, it's quite complicated so see the original paper."""
        means = (self.rewards / self.pulls)
        
        # for monotonize
        # gap = 1/sqrt(self.t)
        gap = 1/(1+100*log_plus(log_plus(self.t)))
        
        if np.any(self.pulls < 1):
            return np.random.choice(np.nonzero(self.pulls < 1)[0])

        eta_solution = self._solve_optimization_problem(means, self.embeddings, **self._kwargs) # should delete zeta
        if np.all(eta_solution == -1):
            eta_t = self.old_mt
        else:
            eta_t = eta_solution
            self.old_mt = eta_t.copy()
        # min{\eta_{n,i}, log(n)}
        eta_t[eta_t > log_plus(self.t)] = log_plus(self.t) 

        LCvalue = np.inf
        if 'L' in self._kwargs and self._kwargs['L'] == -1:
            LCvalue = estimate_Lipschitz_constant(means, self.embeddings)
        elif self._info_on_solver == ", Lipschitz, true":
            LCvalue = self._kwargs['L']
        self.LC_value = LCvalue
        
        # Monotonize
        if np.all(self.pulls[np.where(means == np.max(means))[0]]<(log_plus(self.t)+1)):
            chosen_arm = np.random.choice(np.nonzero(self.pulls == np.min(self.pulls[np.where(means == np.max(means))[0]]))[0])
            return chosen_arm

        # Estimate
        # undersampled_arms = (np.where(self.pulls <= 2*log_plus(self.t)/log_plus(log_plus(self.t))))[0]
        elif (np.where(self.pulls <= 10*log_plus(self.t)/(1+log_plus(log_plus(self.t)))))[0].size > 0:
            # under-sampled arm
            self.phase = Phase.estimation
            chosen_arm = np.random.choice(np.nonzero(self.pulls == np.min(self.pulls[np.where(self.pulls <=10*log_plus(self.t)/(1+log_plus(log_plus(self.t))))[0]]))[0])
            
            return chosen_arm

        elif np.all(self.pulls >= (1+self.gamma) * log_plus(self.t) * eta_t):
            self.phase = Phase.exploitation
            for i in range(self.nbArms):
                if abs(max(means)-means[i])<=gap:
                    means[i] = max(means)
            best_arms = np.where(means == np.max(means))[0]
            chosen_arm = np.random.choice(np.nonzero(self.pulls == np.min(self.pulls[best_arms]))[0])
            return chosen_arm

        else:
            self.phase = Phase.exploration
            for i in range(self.nbArms):
                if abs(max(means)-means[i])<=gap:
                    means[i] = max(means)

            underexplored_arms = (1+self.gamma) * eta_t * log_plus(self.t) - self.pulls
            # most under-explored arm
            chosen_arm = np.random.choice(np.nonzero(underexplored_arms == np.max(underexplored_arms))[0])
            return chosen_arm

class LipschitzOSSB(OSSB_DEL):
    def __init__(self, nbArms, embeddings, gamma=GAMMA, L=-1, **kwargs):
        kwargs.update({'L': L})
        super(LipschitzOSSB, self).__init__(nbArms, embeddings, gamma=gamma, solve_optimization_problem="Lipschitz", LC_value="estimated", **kwargs)

class LipschitzOSSB_DEL_true(OSSB_DEL):
    def __init__(self, nbArms, embeddings, gamma=GAMMA, L=trueLC, **kwargs):
        kwargs.update({'L': L})
        super(LipschitzOSSB_DEL_true, self).__init__(nbArms, embeddings, gamma=gamma, solve_optimization_problem="Lipschitz", LC_value="true", **kwargs)

