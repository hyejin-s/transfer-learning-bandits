# -*- coding: utf-8 -*-

from enum import Enum  # For the different phases
import numpy as np
from scipy.optimize import linprog
from itertools import combinations
from math import log

eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]
# Bernoulli KL-divergence
def klBern(x, y):
   x = min(max(x, eps), 1 - eps)
   y = min(max(y, eps), 1 - eps)
   return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))
klBern_vect = np.vectorize(klBern)

def log_plus(x):
    return max(1, log(x))

#: Different phases during the OSSB algorithm
Phase = Enum('Phase', ['initialisation', 'exploitation', 'estimation', 'exploration'])

#: Default value for the :math:`\gamma` parameter, 0.0 is a safe default.
GAMMA = 0.001
EPSILON = 1

LCvalue = -1    # to print LC

arms = np.array([0.1, 0.0005, 0.0005, 0.2005, 0.0005, 0.0005])
embeddings = [0, 0.995, 0.996, 0.997, 0.998, 0.999]
#embeddings = [0, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999, 1]
#embeddings = [0, 0.99, 0.991, 0.992, 0.993, 0.994, 0.995, 0.996, 0.997, 0.998, 0.999, 1]
#arms = np.array([0.1, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.2, 0.001]) # need to check suboptimal(position)

def estimate_Lipschitz_constant(thetas):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i]-thetas[i+1])/(embeddings[i+1]-embeddings[i]))

    return np.amax(L_values)

# for known Lipschitz Constant
trueLC = estimate_Lipschitz_constant(arms)
wantLC = 0.1

def get_confusing_bandit(k, L, thetas, gap):
    theta_max = np.amax(thetas)
    # values : \lambda_i^k (arm k,i \in K^{-})
    lambda_values = np.zeros_like(thetas)
    for i, theta in enumerate(thetas):
        lambda_values[i] = max(theta, theta_max-L*abs(embeddings[k]-embeddings[i]))
        if L == np.inf:
            if k == i:
                lambda_values[i] = theta_max
        if theta_max-lambda_values[i] <= gap:
            lambda_values[i] = theta_max
    return lambda_values

def solve_optimization_problem__Lipschitz(thetas, gap, L=-1):
    if L==-1:
        tol = 1e-12
    else:
        tol = 1e-8

    theta_max = np.amax(thetas)
    c = theta_max - thetas  # c : (\theta^*-theta_k)_{k\in K}
    
    sub_arms = (np.nonzero(c))[0]
    opt_arms = (np.where(c==0))[0]

    if sub_arms.size==0:    # ex) arms' mean => all 0
        global LCvalue
        LCvalue = 0
        return np.full(thetas.size, np.inf)

    # for unknown Lipschitz Constant
    if L==-1:
        L = estimate_Lipschitz_constant(thetas)
        LCvalue = L #to print LC

    A_ub=np.zeros((sub_arms.size, sub_arms.size))
    for j, k in enumerate(sub_arms):
        nu = get_confusing_bandit(k, L, thetas, gap) # get /lambda^k
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
        res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='revised simplex', bounds=bounds_sub)
    except Exception as e:
        print(str(e))
        res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=bounds_sub)
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

def solve_optimization_problem__classic(thetas):
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
        self.eta_solution = np.zeros(nbArms)
        self.eta_compare = np.zeros(nbArms)
        self.compare_info = np.zeros(3)
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


# Algorithm1
class DEL_bandit(BasePolicy):

    def __init__(self, nbArms, epsilon=EPSILON, gamma=GAMMA,
                 solve_optimization_problem="classic", LC_value=False,
                 lower=0., amplitude=1., **kwargs):
        super(DEL_bandit, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        # Arguments
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
        if solve_optimization_problem == "Lipschitz" and LC_value=="want":
            self._info_on_solver = ", Lipschitz, want"
            self._solve_optimization_problem = solve_optimization_problem__Lipschitz
        self._kwargs = kwargs  # Keep in memory the other arguments, to give to self._solve_optimization_problem

    def __str__(self):
        """ -> str"""
        return r"DEL_bandit($\varepsilon={:.3g}$, $\gamma={:.3g}${})".format(self.epsilon, self.gamma, self._info_on_solver)

    # --- Start game, and receive rewards

    def startGame(self):
        """ Start the game (fill pulls and rewards with 0)."""
        super(DEL_bandit, self).startGame()
        self.counter_s_no_exploitation_phase = 0
        self.phase = Phase.initialisation
        self.compare_info = np.zeros(3)

    def getReward(self, arm, reward):
        """ Give a reward: increase t, pulls, and update cumulated sum of rewards for that arm (normalized in [0, 1])."""
        super(DEL_bandit, self).getReward(arm, reward)

    # --- Basic choice() and handleCollision() method

    def choice(self):
        """ Applies the OSSB procedure, it's quite complicated so see the original paper."""
        means = (self.rewards / self.pulls)
        gap = 1/(1+log_plus(log_plus(self.t)))

        count_undersample = 0
        zeta = np.zeros(self.nbArms)
        zeta_value = self.pulls/log_plus(self.t)
        
        for i in range(self.nbArms):
                if abs(max(means)-means[i])<=gap:
                    means[i] = max(means)

        for i in range(self.nbArms):
            if i in np.where(means == max(means))[0]:
                zeta[i] = np.inf
            else:
                zeta[i] = zeta_value[i]
                count_undersample += 1
        
        global LCvalue
        LCvalue = np.inf
        if 'L' in self._kwargs and self._kwargs['L'] == -1:
            LCvalue = estimate_Lipschitz_constant(means)
        elif 'L' in self._kwargs and self._kwargs['L'] == trueLC:
            LCvalue = trueLC
        elif 'L' in self._kwargs and self._kwargs['L'] == wantLC:
            LCvalue = wantLC
        self.LC_value = LCvalue
        
        check_sum = np.zeros(self.nbArms)
        sum_cons = np.zeros(count_undersample)
        for idx, i in enumerate(np.where(means != max(means))[0]):
            nu_confus = get_confusing_bandit(i, LCvalue, means, gap)
            for k in np.where(means != max(means))[0]:
                sum_cons[idx] += klBern(means[k], nu_confus[k]) * (zeta[k] / (1 + self.gamma))
                check_sum[i] += klBern(means[k], nu_confus[k]) * (zeta[k] / (1 + self.gamma))
        self.zeta_info = check_sum
        
        if np.any(self.pulls < 1):
            return np.random.choice(np.nonzero(self.pulls < 1)[0])

        # Monotonize
        # optimal_arm = np.where(means == np.max(means))[0]
        elif np.all(self.pulls[np.where(means == np.max(means))[0]]<(log_plus(self.t)**2+1)):
            chosen_arm = np.random.choice(np.nonzero(self.pulls == np.min(self.pulls[np.where(means == np.max(means))[0]]))[0])
            return chosen_arm

        # Estimate       
        # underSampledArms = np.where(self.pulls <= log_plus(self.t)/(1+log_plus(log_plus(self.t))))[0]
        elif (np.where(self.pulls <= log_plus(self.t)/(1+log_plus(log_plus(self.t))))[0]).size > 0:
            self.phase = Phase.estimation
            self.compare_info[0] += 1
            chosen_arm = np.random.choice(np.nonzero(self.pulls == np.min(self.pulls[np.where(self.pulls <= log_plus(self.t)/(1+log_plus(log_plus(self.t))))[0]]))[0])
            self.eta_solution = 1
            return chosen_arm   

        # Exploit
        elif np.all(sum_cons >= 1):
            self.phase = Phase.exploitation
            self.compare_info[1] += 1
            for i in range(self.nbArms):
                if abs(max(means)-means[i])<=gap:
                    means[i] = max(means)
            bestvalue_arm = np.where(means == np.max(means))[0]
            chosen_arm = np.random.choice(np.nonzero(self.pulls == np.min(self.pulls[bestvalue_arm]))[0])
            self.eta_solution = 2
            return chosen_arm

        else:
            # for error
            for i in range(self.nbArms):
                if abs(max(means)-means[i])<=gap:
                    means[i] = max(means)
            thistime_mt = self._solve_optimization_problem(means, **self._kwargs)
            if np.all(thistime_mt == -1):
                values_c_x_mt = self.old_mt
            else:
                values_c_x_mt = thistime_mt
                self.oldmt = values_c_x_mt.copy()
            self.eta_solution = values_c_x_mt.copy()
   
            values_c_x_mt2 = np.zeros(self.nbArms)
            for i in range(self.nbArms):
                if i in np.where(means != max(means))[0]:
                    values_c_x_mt2[i] = min((1+self.gamma)*values_c_x_mt[i], log_plus(self.t))
                else:
                    values_c_x_mt2[i] = log_plus(self.t)
            self.eta_compare = values_c_x_mt2.copy()

            # exploration          
            self.phase = Phase.exploration
            self.compare_info[2] += 1
            # most under-explored arm
            values = values_c_x_mt2 * log_plus(self.t) - self.pulls
            chosen_arm = np.random.choice(np.nonzero(values == np.max(values))[0])
            return chosen_arm


# class LipschitzOSSB_DEL(OSSB_DEL):
#     def __init__(self, nbArms, gamma=GAMMA, L=-1, **kwargs):
#         kwargs.update({'L': L})
#         super(LipschitzOSSB_DEL, self).__init__(nbArms, gamma=gamma, solve_optimization_problem="Lipschitz", LC_value="estimated", **kwargs)

# class LipschitzOSSB_DEL_true(OSSB_DEL):
#     def __init__(self, nbArms, gamma=GAMMA, L=trueLC, **kwargs):
#         kwargs.update({'L': L})
#         super(LipschitzOSSB_DEL_true, self).__init__(nbArms, gamma=gamma, solve_optimization_problem="Lipschitz", LC_value="true", **kwargs)

# class LipschitzOSSB_DEL_want(OSSB_DEL):
#     def __init__(self, nbArms, gamma=GAMMA, L=wantLC, **kwargs):
#         kwargs.update({'L': L})
#         super(LipschitzOSSB_DEL_want, self).__init__(nbArms, gamma=gamma, solve_optimization_problem="Lipschitz", LC_value="want", **kwargs)