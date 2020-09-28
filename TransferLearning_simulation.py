from Bernolli import Bernoulli
from evaluator_transferlearning import *
from OSSB_Transfer import OSSB_DEL, LipschitzOSSB_DEL, LipschitzOSSB_DEL_beta, LipschitzOSSB_DEL_true

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

M = 5 # the ordering of Transfer Learning
delta_gap = minimal_gap(embeddings)
alpha = 0.2
beta = 0.1
epsilon = 0.001
# self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)


from GenerativeModel import GenerativeModel, embeddings
generation = GenerativeModel(10) #The number of arms = 10

regret_transfer = []
regret_estimated = []
regret_true = []

##### Transfer Learning #####
Empirical_Lipschitz = []    # store empirical Lipschitz constant every episode
POLICIES = [{"archtype":OSSB_DEL, "params":{}}, {"archtype":LipschitzOSSB_DEL, "params":{}}]   
for m in range(M):
    print("{} episode".format(m+1))
    arms = generation.gene_arms(5)    # Lipschitz Constant = 5
    ENVIRONMENTS = [{"arm_type": Bernoulli, "params": arms}]

    ##### Transfer Learning #####
    if m == 0:   # First of all, run OSSB     
        # OSSB_DEL
        configuration_beta = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 6, "environment": ENVIRONMENTS, "policies": [POLICIES[0]]}  
    else:
        tau = np.min(evaluation_beta.lastPulls[0][0][:,0]) # The smallest number of pulls (just before?)
        left_cal = (delta_gap**2)*(epsilon**2)*min(alpha, (alpha-beta))*tau*M
        right_cal = log(HORIZON)
        if left_cal > right_cal: #L_beta
            betaLC = Lipschitz_beta(Empirical_Lipschitz, beta, m)
            # OSSB_beta
            configuration_beta = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 6, "environment": ENVIRONMENTS,
            "policies": [LipschitzOSSB_DEL_beta(nbArms=10, gamma=0.001, L=betaLC)]}   
        else:
            # OSSB_DEL
            configuration_beta = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 6, "environment": ENVIRONMENTS, "policies": [POLICIES[0]]}  

    evaluation_beta = Evaluator(configuration_beta)
    for envId, env in tqdm(enumerate(evaluation_beta.envs), desc="Problems"):
        evaluation_beta.startOneEnv(envId, env)
    ## evaluation.get_lastmeans()[0][0] -> [ nbArms ]
    Empirical_Lipschitz.append(estimate_Lipschitz_constant(evaluation_beta.get_lastmeans()[0][0]))
    regret_transfer.append(evaluation_beta.getCumulatedRegret_MoreAccurate(policyId=0)[HORIZON-1])

    ##### estimated #####
    configuration_estimated = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 6, "environment": ENVIRONMENTS, "policies": [POLICIES[1]]}
    evaluation_est = Evaluator(configuration_estimated)
    for envId, env in tqdm(enumerate(evaluation_est.envs), desc="Problems"):
        evaluation_est.startOneEnv(envId, env)
    regret_estimated.append(evaluation_est.getCumulatedRegret_MoreAccurate(policyId=0)[HORIZON-1])
    
    ##### true #####
    trueLC = estimate_Lipschitz_constant(arms) 
    configuration_true = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 6, "environment": ENVIRONMENTS,
    "policies": [LipschitzOSSB_DEL_true(nbArms=10, gamma=0.001, L=trueLC)]}
    evaluation = Evaluator(configuration_true)
    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
        evaluation.startOneEnv(envId, env)
    regret_true.append(evaluation.getCumulatedRegret_MoreAccurate(policyId=0)[HORIZON-1])


colors = ['tomato', 'limegreen', 'deepskyblue']

def plotRegret(filepath):
    plt.figure()
    X = list(range(1,M+1))
    plt.plot(X, regret_transfer, label="Transfer Learning", color = colors[0])
    plt.plot(X, regret_estimated, label="Estimated Lipschitz", color = colors[1])
    plt.plot(X, regret_true, label="True Lipschitz", color = colors[2])
    legend()
    plt.savefig(filepath+'/Regret', dpi=300)

dir_name = "simulation_transfer/test"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

plotRegret(dir_name)
