from Bernolli import Bernoulli
from evaluator_transferlearning import *
from OSSB_Transfer import OSSB_DEL, LipschitzOSSB_DEL, LipschitzOSSB_DEL_beta, LipschitzOSSB_DEL_true

import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log, ceil
import os
import numpy as np

HORIZON=10000
REPETITIONS=1
N_JOBS=40



M = 400 # the ordering of Transfer Learning
bound = 100 # num of L = inf
geneLC = 0.2

#dir_name = "7_estimate*1"
#if not os.path.exists(dir_name):
#    os.makedirs(dir_name)

from GenerativeModel import GenerativeModel, embeddings
generation = GenerativeModel(10) #The number of arms = 10


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

def com_Lipschitz_constant(thetas):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i+1]-thetas[i])/(embeddings[i+1]-embeddings[i]))

    return np.amax(L_values)
print(embeddings)

valuelist = [[0.05, 0.05], [0.2, 0.05], [0.05, 0.5], [0.2, 0.5]] # [beta, epsilon]
regret_inf = np.zeros(M)
regret_true = np.zeros(M)
regret_estimated = np.zeros(M)
regret_transfer = np.zeros((4,M))

# for print
arm_info = np.zeros((M,10))
beta_info = np.zeros((4,M))
com_info = np.zeros((4,M,2))
trueLC_list = []
lastpull = np.zeros((M,7,10)) #(M, nbPolicies, numArms)

##### Transfer Learning #####
Empirical_Lipschitz = np.zeros((7,M))   # store empirical Lipschitz constant every episode
# POLICIES = [{"archtype":OSSB_DEL, "params":{}}, {"archtype":LipschitzOSSB_DEL, "params":{}}]   
for m in range(M):
    POLICIES = []
    print("{} episode".format(m+1))
    # Generate model
    arms = generation.gene_arms(geneLC)  # Lipschitz Constant = 0.2
    arm_info[m] = arms

    ENVIRONMENTS = [{"arm_type": Bernoulli, "params": arms}]
    
    ##### Transfer Learning #####
    for idx, value in enumerate(valuelist):
        beta = value[0]
        epsilon = value[1]

        if m < bound :   # First of all, run OSSB   
            # OSSB_DEL
            POLICIES.append({"archtype":OSSB_DEL, "params":{}})
        else:
            betaLC = Lipschitz_beta(Empirical_Lipschitz[idx], epsilon, beta, m)
            beta_info[idx][m] = betaLC
            # OSSB_beta
            POLICIES.append(LipschitzOSSB_DEL_beta(nbArms=10, gamma=0.001, L=betaLC))    


    ##### inf #####
    POLICIES.append({"archtype":OSSB_DEL, "params":{}})

    ##### estimated #####
    POLICIES.append({"archtype":LipschitzOSSB_DEL, "params":{}}) 

    ##### true #####
    trueLC = com_Lipschitz_constant(arms)
    POLICIES.append(LipschitzOSSB_DEL_true(nbArms=10, gamma=0.001, L=trueLC))
    trueLC_list.append(trueLC)

    configuration = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
    evaluation = Evaluator(configuration)
    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
        evaluation.startOneEnv(envId, env)
    for idx in range(len(valuelist)):
        regret_transfer[idx][m] = evaluation.getCumulatedRegret_LessAccurate(policyId=idx)[HORIZON-1]
    for idx in range(len(POLICIES)):
        Empirical_Lipschitz[idx][m] = com_Lipschitz_constant(evaluation.get_lastmeans()[idx][0])
        
    regret_inf[m] = evaluation.getCumulatedRegret_LessAccurate(policyId=4)[HORIZON-1]
    regret_estimated[m] = evaluation.getCumulatedRegret_LessAccurate(policyId=5)[HORIZON-1]
    regret_true[m] = evaluation.getCumulatedRegret_LessAccurate(policyId=6)[HORIZON-1]

    # self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)
    for idx in range(len(POLICIES)):
        lastpull[m][idx] = evaluation.getLastPulls()[idx, :, 0]
    #evaluation.estimatedLipschitzdata(filepath=dir_name)


colors = ['tomato', 'limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple'] # 7

def plotRegret(filepath):
    plt.figure()
    X = list(range(1, M+1))
    plt.plot(X, regret_inf, label="Inf", color = colors[0])
    plt.plot(X, regret_estimated, label="Estimated Lipschitz", color = colors[1])
    plt.plot(X, regret_true, label="True Lipschitz", color = colors[2])
    for i in range(len(valuelist)):
        plt.plot(X, regret_transfer[i], label="Transfer Learning{}:(beta, epsilon)=({}, {})".format(i+1, valuelist[i][0], valuelist[i][1]), color=colors[i+3])

    plt.legend()
    plt.title("Total {} episode, {} horizon".format(M, HORIZON))
    plt.savefig(filepath+'/Regret', dpi=300)

with open(dir_name+ "/L_beta.txt", "w") as f:
    for i in range(len(beta_info)):
        f.write("\n{}".format(i))
        for episode in range(len(beta_info[i])):
            f.write("\nepisode:{}, {}".format(episode, beta_info[i, episode]))

with open(dir_name+ "/trueLC.txt", "w") as f:
    for i in range(len(trueLC_list)):
        f.write("episode:{}, {}\n".format(i, trueLC_list[i]))

with open(dir_name+ "/EmpiricalLipschitz.txt", "w") as f:
    for i in range(len(Empirical_Lipschitz)):
        f.write("\n{}".format(i))
        for episode in range(len(Empirical_Lipschitz[i])):
            f.write("\nepisode:{}, {}".format(episode, Empirical_Lipschitz[i, episode]))

with open(dir_name+ "/regret_transfer.txt", "w") as f:
    for episode in range(M): 
        f.write("\nepisode:{}, {}, {}, {}, {}".format(episode, regret_transfer[0, episode], regret_transfer[1, episode], regret_transfer[2, episode], regret_transfer[3, episode]))

with open(dir_name+ "/regret_info.txt", "w") as f:
    for episode in range(M):
        f.write("\nepisode:{}, {}, {}, {}".format(episode, regret_inf[episode], regret_true[episode], regret_estimated[episode]))

with open(dir_name+ "/arm_info.txt", "w") as f:
    for i in range(M):
        f.write("\nepisode:{}, {}".format(i, arm_info[i]))

#lastpull = np.zeros((M,7,10)) #(M, nbPolicies, numArms)
with open(dir_name+ "/lastpull.txt", "w") as f:
    for m in range(M):
         f.write("\nepisode:{}".format(m))
         for policyId in range(7):
             f.write("\npolicy:{}\n".format(policyId))
             np.savetxt(f, lastpull[m][policyId].astype(int), fmt='%i', newline=", ")


with open(dir_name+ "/information.txt", "w") as f:
    f.write("Number of episodes: " + str(M))
    f.write("\nHorizon: " + str(HORIZON))
    f.write("\nbound:" + str(bound))
    f.write("\n(beta, epsilon): " + str(valuelist))
    f.write("\nGeneration Lipschitz Constant: "+ str(geneLC))
    f.write("\nembeddings: "+str(embeddings))

plotRegret(dir_name)
