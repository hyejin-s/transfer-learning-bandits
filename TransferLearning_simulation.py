from Bernolli import Bernoulli
from evaluator_transferlearning import *
from OSSB_Transfer import OSSB_DEL, LipschitzOSSB_DEL, LipschitzOSSB_DEL_beta, LipschitzOSSB_DEL_true

import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log, ceil
import os
import numpy as np

def minimal_gap(embed):
    gap_list = []
    for i in range(len(embed)-1):
        gap_list.append(embed[i+1]-embed[i])

    return np.min(gap_list)

def Lipschitz_beta(L_list, epsilon, beta, M_present):
    bound_value = ceil(beta*M_present) # ROUND UP
    #L_list.sort(reverse=True)
    L_beta = sorted(L_list, reverse=True)[bound_value-1] + epsilon*beta
    return L_beta

HORIZON=10
REPETITIONS=1 
N_JOBS=1

M = 2 # the ordering of Transfer Learning
delta_gap = minimal_gap(embeddings)

alpha = 0.3
"""
beta = 0.1
epsilon = 0.001
"""
# self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)


from GenerativeModel import GenerativeModel, embeddings
generation = GenerativeModel(10) #The number of arms = 10

valuelist = [[0.05, 0.1], [0.2, 0.1], [0.05, 0.5], [0.2, 0.5]] # [beta, epsilon]
regret_inf = np.zeros(M)
regret_true = np.zeros(M)
regret_estimated = np.zeros(M)
regret_transfer = np.zeros((4,M))

# for print
beta_info = np.zeros((4,M))
com_info = np.zeros((4,M,2))
tau_list = []

##### Transfer Learning #####
Empirical_Lipschitz = np.zeros((4,M))   # store empirical Lipschitz constant every episode
# POLICIES = [{"archtype":OSSB_DEL, "params":{}}, {"archtype":LipschitzOSSB_DEL, "params":{}}]   
for m in range(M):
    POLICIES = []
    print("{} episode".format(m+1))
    # Generate model
    arms = generation.gene_arms(0.2)    # Lipschitz Constant = 0.2
    ENVIRONMENTS = [{"arm_type": Bernoulli, "params": arms}]
    for idx, value in enumerate(valuelist):
        ##### Transfer Learning #####
        if m < 2 :   # First of all, run OSSB   
            # OSSB_DEL
            POLICIES.append({"archtype":OSSB_DEL, "params":{}})
        else:
            beta = value[0]
            epsilon = value[1]
            tau_list.append(np.min(evaluation.lastPulls[0][0][:,0])) # The smallest number of pulls (just before)
            tau = min(tau_list)
            left_cal = (delta_gap**2)*(epsilon**2)*min(alpha, (alpha-beta))*tau*M
            right_cal = log(HORIZON)
            
            com_info[idx][m][0] = left_cal
            com_info[idx][m][1] = right_cal
            if left_cal > right_cal: #L_beta
                betaLC = Lipschitz_beta(Empirical_Lipschitz[idx], epsilon, beta, m)
                beta_info[idx][m] = betaLC
                # OSSB_beta
                POLICIES.append(LipschitzOSSB_DEL_beta(nbArms=10, gamma=0.001, L=betaLC))
            else:
                # OSSB_DEL
                POLICIES.append({"archtype":OSSB_DEL, "params":{}})

    ##### inf #####
    POLICIES.append({"archtype":OSSB_DEL, "params":{}})

    ##### estimated #####
    POLICIES.append({"archtype":LipschitzOSSB_DEL, "params":{}}) 

    ##### true #####
    trueLC = estimate_Lipschitz_constant(arms) 
    POLICIES.append(LipschitzOSSB_DEL_beta(nbArms=10, gamma=0.001, L=trueLC))


    configuration = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 6, "environment": ENVIRONMENTS, "policies": POLICIES}
    evaluation = Evaluator(configuration)
    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
        evaluation.startOneEnv(envId, env)
    
    for idx, value in enumerate(valuelist):
        Empirical_Lipschitz[idx][m] = estimate_Lipschitz_constant(evaluation.get_lastmeans()[idx][0])
        regret_transfer[idx][m] = evaluation.getCumulatedRegret_MoreAccurate(policyId=idx)[HORIZON-1]
   
    regret_inf[m] = evaluation.getCumulatedRegret_MoreAccurate(policyId=4)[HORIZON-1]
    regret_estimated[m] = evaluation.getCumulatedRegret_MoreAccurate(policyId=5)[HORIZON-1]
    regret_true[m] = evaluation.getCumulatedRegret_MoreAccurate(policyId=6)[HORIZON-1]


colors = ['tomato', 'limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple'] # 7

def plotRegret(filepath):
    plt.figure()
    X = list(range(1, M+1))
    plt.plot(X, regret_inf, label="Inf", color = colors[0])
    plt.plot(X, regret_estimated, label="Estimated Lipschitz", color = colors[1])
    plt.plot(X, regret_true, label="True Lipschitz", color = colors[2])
    for i in range(len(valuelist)):
        plt.plot(X, regret_transfer[i], label="Transfer Learning{}:(beta, epsilon)=({}, {})".format(i+1, valuelist[i][0], valuelist[i][1]), color=colors[i+3])

    legend()
    plt.title("Total {} episode, {} horizon".format(M, HORIZON))
    plt.savefig(filepath+'/Regret', dpi=300)

dir_name = "simulation_transfer/test"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

with open(dir_name+ "/L_beta.txt", "w") as f:
    for i in range(len(beta_info)):
        f.write("\n{}".format(i))
        for episode in range(len(beta_info[i])):
            f.write("\nepisode:{}, {}".format(episode, beta_info[i, episode]))

with open(dir_name+ "/EmpiricalLipschitz.txt", "w") as f:
    for i in range(len(Empirical_Lipschitz)):
        f.write("\n{}".format(i))
        for episode in range(len(Empirical_Lipschitz[i])):
            f.write("\nepisode:{}, {}".format(episode, Empirical_Lipschitz[i, episode]))
with open(dir_name+ "/regret_transfer.txt", "w") as f:
    for i in range(len(regret_transfer)):
        f.write("\n{}".format(i))
        for episode in range(len(regret_transfer[i])):
            f.write("\nepisode:{}, {}".format(episode, regret_transfer[i, episode]))

with open(dir_name+ "/com_info.txt", "w") as f:
    for i in range(len(com_info)):
        f.write("\n{}".format(i))
        for episode in range(len(com_info[i])):
            f.write("\nepisode:{}, {}, {}".format(episode, com_info[i, episode, 0], com_info[i, episode, 1]))

plotRegret(dir_name)
