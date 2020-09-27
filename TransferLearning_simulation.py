from Bernolli import Bernoulli
from evaluator import *
from OSSB_Transfer import OSSB, OSSB_DEL, LipschitzOSSB_DEL, LipschitzOSSB_DEL_beta, arms, embeddings

import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log, ceil
import os
import numpy as npS

def minimal_gap(embed):
    gap_list = []
    for i in range(len(embed)-1):
        gap_list.append(embed[i+1]-embed[i])
    
    return np.min(gap_list)

def Lipschitz_beta(L_list, beta, M_present):
    bound_value = ceil(beta*M_present) # ROUND UP
    print(L_list)
    print(bound_value)
    #L_list.sort(reverse=True)
    L_beta = sorted(L_list, reverse=True)[bound_value-1]
    return L_beta


HORIZON=1000
REPETITIONS=1 
N_JOBS=1

### Transfer Learning ###

# First of all, run OSSB 
ENVIRONMENTS = [{"arm_type": Bernoulli, "params": arms}]
POLICIES = [{"archtype":OSSB_DEL, "params":{}}]      

configuration_basic = {
    "horizon": HORIZON,
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Arms
    "environment": ENVIRONMENTS,
    # --- Algorithms
    "policies": POLICIES,    # OSSB_DEL
}

Empirical_Lipschitz = [] # store empirical Lipschitz constant every episode
Cumulative_reward = []

# First of all, run OSSB
evaluation = Evaluator(configuration_basic)
for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
    evaluation.startOneEnv(envId, env)

## evaluation.get_lastmeans()[0][0] -> [ nbArms ]
Empirical_Lipschitz.append(estimate_Lipschitz_constant(evaluation.get_lastmeans()[0][0]))
Cumulative_reward.append(evaluation.getCumulatedRegret_MoreAccurate(policyId=0)[HORIZON-1])

M = 10 # the ordering of Transfer Learning
delta_gap = minimal_gap(embeddings)
alpha = 0.2
beta = 0.1
epsilon = 0.001
# self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)
tau = np.min(evaluation.lastPulls[0][0][:,0]) # The smallest number of pulls

left_cal = (delta_gap**2)*(epsilon**2)*min(alpha, (alpha-beta))*tau*M
right_cal = log(HORIZON)

for m in range(M-1):
    if left_cal > right_cal: #L_beta
        betaLC = Lipschitz_beta(Empirical_Lipschitz, beta, m+1)

        configuration_beta = {
        "horizon": HORIZON,
        "repetitions": REPETITIONS,
        # --- Parameters for the use of joblib.Parallel
        "n_jobs": N_JOBS,    # = nb of CPU cores
        "verbosity": 6,      # Max joblib verbosity
        # --- Arms
        "environment": ENVIRONMENTS,
        # --- Algorithms
        "policies": [LipschitzOSSB_DEL_beta(nbArms=10, gamma=0.001, L=betaLC)],    # OSSB_beta
    }   
        evaluation = Evaluator(configuration_beta)
        for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
            evaluation.startOneEnv(envId, env)

        Empirical_Lipschitz.append(estimate_Lipschitz_constant(evaluation.get_lastmeans()[0][0]))
        Cumulative_reward.append(evaluation.getCumulatedRegret_MoreAccurate(policyId=0)[HORIZON-1])

    else:
        evaluation = Evaluator(configuration_basic)
        for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
            evaluation.startOneEnv(envId, env)

        Empirical_Lipschitz.append(estimate_Lipschitz_constant(evaluation.get_lastmeans()[0][0]))
        Cumulative_reward.append(evaluation.getCumulatedRegret_MoreAccurate(policyId=0)[HORIZON-1])

plt.figure()
X = list(range(1,M+1))
plt.plot(X, Cumulative_reward)
plt.show()


### Known Lipschitz Constant ###


### Unknown(estimated) Lipschitz Constant ###

dir_name = "simulation_transfer/test"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)
