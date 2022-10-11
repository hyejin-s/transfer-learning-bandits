from Bernolli import Bernoulli
from evaluator_transferlearning import *
from OSSB_Transfer10 import OSSB_DEL, LipschitzOSSB_DEL, LipschitzOSSB_DEL_beta, LipschitzOSSB_DEL_true

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
geneLC = 1

dir_name = "last_max1"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

from GenerativeModel6 import GenerativeModel, embeddings
generation = GenerativeModel(6) #The number of arms = 6


def Lipschitz_beta(L_list, epsilon, beta, M_present):
    bound_value = ceil(beta*M_present) # ROUND UP
    #L_list.sort(reverse=True)
    L_beta = sorted(L_list, reverse=True)[bound_value-1] + epsilon*beta
    return L_beta

def com_Lipschitz_constant(thetas, embeddings):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i+1]-thetas[i])/(embeddings[i+1]-embeddings[i]))

    return np.amax(L_values)


valuelist = [[0.1, 0.05], [0.1, 0.1]] # [beta, epsilon]
regret_inf = np.zeros(M)
regret_true = np.zeros(M)
regret_estimated = np.zeros(M)
regret_transfer = np.zeros((6,M))

# for print
arm_info = np.zeros((M,6))
beta_info = np.zeros((2,M))
max_info = np.zeros((2,M))

trueLC_list = []
lastpull = np.zeros((M,7,6)) #(M, nbPolicies, numArms)
lastmean = np.zeros((M,7,6))

##### Transfer Learning #####
Empirical_Lipschitz = np.zeros((7,M))   # store empirical Lipschitz constant every episode
# POLICIES = [{"archtype":OSSB_DEL, "params":{}}, {"archtype":LipschitzOSSB_DEL, "params":{}}]   
for m in range(M):
    POLICIES = []
    POLICIES2 = []
    print("{} episode".format(m+1))
    # Generate model
    arms = generation.gene_arms(geneLC)  # Lipschitz Constant = 0.2
    arm_info[m] = arms

    ENVIRONMENTS = [{"arm_type": Bernoulli, "params": arms}]
    
    ##### Transfer Learning #####
    ##### Transfer Max Learning #####
    for idx, value in enumerate(valuelist):
        beta = value[0]
        epsilon = value[1]

        if m < bound :   # First of all, run OSSB   
            # OSSB_DEL
            POLICIES.append({"archtype":OSSB_DEL, "params":{}})
            POLICIES.append({"archtype":OSSB_DEL, "params":{}})
            POLICIES2.append({"archtype":OSSB_DEL, "params":{}})
        else:
            betaLC = Lipschitz_beta(Empirical_Lipschitz[idx*2], epsilon, beta, m)
            beta_info[idx][m] = betaLC
            # OSSB_beta
            POLICIES.append(LipschitzOSSB_DEL_beta(nbArms=6, gamma=0.001, L=betaLC))
            POLICIES2.append(LipschitzOSSB_DEL_beta(nbArms=6, gamma=0.001, L=betaLC))
            maxLC = max(Empirical_Lipschitz[idx*2+1])
            max_info[idx][m] = maxLC
            POLICIES.append(LipschitzOSSB_DEL_beta(nbArms=6, gamma=0.001, L=maxLC))

    ##### inf #####
    #POLICIES.append({"archtype":OSSB_DEL, "params":{}})
    ##### estimated #####
    #POLICIES.append({"archtype":LipschitzOSSB_DEL, "params":{}}) 

    ##### true #####
    trueLC = com_Lipschitz_constant(arms, embeddings)
    POLICIES.append(LipschitzOSSB_DEL_true(nbArms=6, gamma=0.001, L=trueLC))
    trueLC_list.append(trueLC)

    configuration = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
    evaluation = Evaluator(configuration)

    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
        evaluation.startOneEnv(envId, env)
    for idx in range(4):
        regret_transfer[idx][m] = evaluation.getCumulatedRegret_LessAccurate(policyId=idx)[HORIZON-1]
    for idx in range(len(POLICIES)):
        Empirical_Lipschitz[idx][m] = com_Lipschitz_constant(evaluation.get_lastmeans()[idx][0], embeddings)

    #regret_inf[m] = evaluation.getCumulatedRegret_LessAccurate(policyId=4)[HORIZON-1]
    #regret_estimated[m] = evaluation.getCumulatedRegret_LessAccurate(policyId=5)[HORIZON-1]
    regret_true[m] = evaluation.getCumulatedRegret_LessAccurate(policyId=4)[HORIZON-1]

    # self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)
    for idx in range(len(POLICIES)):
        lastpull[m][idx] = evaluation.getLastPulls()[idx, :, 0]
    for idx in range(len(POLICIES)):
        lastmean[m][idx] = evaluation.get_lastmeans()[idx][0]
    #evaluation.estimatedLipschitzdata(filepath=dir_name)

    configuration2 = {"horizon": 20000, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES2}
    evaluation2 = Evaluator(configuration2)
    for envId, env in tqdm(enumerate(evaluation2.envs), desc="Problems"):
        evaluation2.startOneEnv(envId, env)
    regret_transfer[4][m] = evaluation2.getCumulatedRegret_LessAccurate(policyId=0)[HORIZON-1]
    Empirical_Lipschitz[5][m] = com_Lipschitz_constant(evaluation2.get_lastmeans()[0][0], embeddings)
    regret_transfer[5][m] = evaluation2.getCumulatedRegret_LessAccurate(policyId=1)[HORIZON-1]
    Empirical_Lipschitz[6][m] = com_Lipschitz_constant(evaluation2.get_lastmeans()[1][0], embeddings)

    lastpull[m][5] = evaluation2.getLastPulls()[0, :, 0]
    lastmean[m][5] = evaluation2.get_lastmeans()[0][0]
    lastpull[m][6] = evaluation2.getLastPulls()[1, :, 0]
    lastmean[m][6] = evaluation2.get_lastmeans()[1][0]
    #evaluation.estimatedLipschitzdata(filepath=dir_name)

colors = ['tomato', 'limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple'] # 7

def plotRegret(filepath):
    plt.figure()
    X = list(range(1, M+1))
    #plt.plot(X, regret_inf, label="Inf", color = colors[0])
    #plt.plot(X, regret_estimated, label="Estimated Lipschitz", color = colors[1])
    plt.plot(X, regret_true, label="True Lipschitz", color = colors[4])
    for i in range(len(valuelist)):
        plt.plot(X, regret_transfer[i*2], label="Transfer Learning{}:(beta, epsilon)=({}, {})".format(i+1, valuelist[i][0], valuelist[i][1]), color=colors[i*2])
        plt.plot(X, regret_transfer[i*2+1], label="Transfer Learning_max{}:(beta, epsilon)=({}, {})".format(i+1, valuelist[i][0], valuelist[i][1]), color=colors[i*2+1])
    plt.plot(X, regret_transfer[4], label="Transfer Learning_max{}:(beta, epsilon)=({}, {})".format(1, valuelist[0][0], valuelist[0][1]), color=colors[5])
    plt.plot(X, regret_transfer[5], label="Transfer Learning_max{}:(beta, epsilon)=({}, {})".format(2, valuelist[1][0], valuelist[1][1]), color=colors[6])
    plt.legend()
    plt.title("Total {} episode, {} horizon".format(M, HORIZON))
    plt.savefig(filepath+'/Regret', dpi=300)

with open(dir_name+ "/L_beta.txt", "w") as f:
    for i in range(len(beta_info)):
        f.write("\n{}".format(i))
        for episode in range(len(beta_info[i])):
            f.write("\nepisode:{}, {}".format(episode, beta_info[i, episode]))

with open(dir_name+ "/L_max.txt", "w") as f:
    for i in range(len(max_info)):
        f.write("\n{}".format(i))
        for episode in range(len(max_info[i])):
            f.write("\nepisode:{}, {}".format(episode, max_info[i, episode]))

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
        f.write("\nepisode:{}, {}, {}, {}, {}, {}, {}".format(episode, regret_transfer[0, episode], regret_transfer[1, episode], regret_transfer[2, episode], regret_transfer[3, episode], regret_transfer[4, episode], regret_transfer[5, episode]))

with open(dir_name+ "/regret_info.txt", "w") as f:
    for episode in range(M):
        f.write("\nepisode:{}, {}".format(episode, regret_true[episode]))

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

with open(dir_name+ "/lastmean.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}".format(m))
        for policyId in range(7):
            f.write("\npolicy:{}\n".format(policyId))
            np.savetxt(f, lastmean[m][policyId], newline=", ", fmt='%1.3f')

with open(dir_name+ "/information.txt", "w") as f:
    f.write("Number of episodes: " + str(M))
    f.write("\nHorizon: " + str(HORIZON))
    f.write("\nbound:" + str(bound))
    f.write("\n(beta, epsilon): " + str(valuelist))
    f.write("\nGeneration Lipschitz Constant: "+ str(geneLC))
    f.write("\nembeddings: "+str(embeddings))

plotRegret(dir_name)
