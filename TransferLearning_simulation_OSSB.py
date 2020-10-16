# for extracting Empirical Lipschitz Constant by running OSSB (no structure)

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
geneLC = 0.5

dir_name = "ha01"
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

#valuelist = [[0.5, 0.05], [0.5, 0.1], [0.1, 0.05], [0.1, 0.1]] # [beta, epsilon]
regret_inf = np.zeros(M)
#regret_true = np.zeros(M)
#regret_estimated = np.zeros(M)
#regret_transfer = np.zeros((4,M))

# for print
arm_info = np.zeros((M,6))
lastpull = np.zeros((M,6)) #(M, numArms)
lastmean = np.zeros((M,6))

trueLC_list = []

##### Transfer Learning #####
Empirical_Lipschitz = np.zeros(M)   # store empirical Lipschitz constant every episode
# POLICIES = [{"archtype":OSSB_DEL, "params":{}}, {"archtype":LipschitzOSSB_DEL, "params":{}}]   
for m in range(M):
    POLICIES = []
    print("{} episode".format(m+1))
    # Generate model
    arms = generation.gene_arms(geneLC)  # Lipschitz Constant = 0.2
    arm_info[m] = arms
    
    ##### true #####
    trueLC = com_Lipschitz_constant(arms, embeddings)
    trueLC_list.append(trueLC)

    ENVIRONMENTS = [{"arm_type": Bernoulli, "params": arms}]
    
    POLICIES.append({"archtype":OSSB_DEL, "params":{}})

    configuration = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
    evaluation = Evaluator(configuration)
    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
        evaluation.startOneEnv(envId, env)
    for idx in range(len(POLICIES)):
        Empirical_Lipschitz[m] = com_Lipschitz_constant(evaluation.get_lastmeans()[idx][0], embeddings)

    regret_inf[m] = evaluation.getCumulatedRegret_LessAccurate(policyId=0)[HORIZON-1]

    # self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)
    for idx in range(len(POLICIES)):
        lastpull[m] = evaluation.getLastPulls()[idx, :, 0]
    for idx in range(len(POLICIES)):
        lastmean[m] = evaluation.get_lastmeans()[idx][0]
    #evaluation.estimatedLipschitzdata(filepath=dir_name)


colors = ['tomato', 'limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple'] # 7


with open(dir_name+ "/EmpiricalLipschitz.txt", "w") as f:
    for i in range(len(Empirical_Lipschitz)):
        f.write("\nepisode:{}, {}".format(i, Empirical_Lipschitz[i]))

with open(dir_name+ "/regret_info.txt", "w") as f:
    for episode in range(M):
        f.write("\nepisode:{}, {}".format(episode, regret_inf[episode]))

with open(dir_name+ "/arm_info.txt", "w") as f:
    for i in range(M):
        f.write("\nepisode:{}, {}".format(i, arm_info[i]))

#lastpull = np.zeros((M,7,10)) #(M, nbPolicies, numArms)
with open(dir_name+ "/lastpull.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}, ".format(m))
        np.savetxt(f, lastpull[m].astype(int), fmt='%i', newline=", ")

with open(dir_name+ "/lastmean.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}, ".format(m))
        np.savetxt(f, lastmean[m], newline=", ", fmt='%1.3f')

with open(dir_name+ "/information.txt", "w") as f:
    f.write("Number of episodes: " + str(M))
    f.write("\nHorizon: " + str(HORIZON))
    f.write("\nbound:" + str(bound))

    f.write("\nGeneration Lipschitz Constant: "+ str(geneLC))
    f.write("\nembeddings: "+str(embeddings))

with open(dir_name+ "/trueLC.txt", "w") as f:
    for i in range(len(trueLC_list)):
        f.write("episode:{}, {}\n".format(i, trueLC_list[i]))