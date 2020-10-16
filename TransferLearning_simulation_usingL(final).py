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

dir_name = "last_regret_bbbbbbb"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

valuelist = [[0.1, 0.05], [0.3, 0.05], [0.5, 0.05]] # [beta, epsilon]
L_info = np.zeros((4,M))


def Lipschitz_beta(L_list, epsilon, beta, M_present):
    bound_value = ceil(beta*M_present) # ROUND UP
    #L_list.sort(reverse=True)
    L_beta = sorted(L_list, reverse=True)[bound_value-1] + epsilon
    return L_beta

def com_Lipschitz_constant(thetas, embeddings):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i+1]-thetas[i])/(embeddings[i+1]-embeddings[i]))

    return np.amax(L_values)

with open(r"/home/phj/TransferLearning/TransferSimulation/ha01/EmpiricalLipschitz.txt") as f:
    data = f.read().split()
    for i in range(len(valuelist)+1):
        Lip_list = []
        Empirical_list = []
        if i == 3:
            for m in range(M):
                Empirical_list.append(float(data[m*2+1]))
                L_info[i][m] = max(Empirical_list)
        else:
            beta = valuelist[i][0]
            epsilon = valuelist[i][1]
            for m in range(M):
                Empirical_list.append(float(data[m*2+1]))
                L_info[i][m] = Lipschitz_beta(Empirical_list, epsilon, beta, m)

embeddings = [0, 0.8, 0.85, 0.9, 0.95, 1]
arm_info = np.zeros((M,6))
with open(r"/home/phj/TransferLearning/TransferSimulation/ha01/arm_info.txt") as f:
    data = f.read().split()
    for m in range(M):
        for idx in range(6):
            if idx == 0:
                arm_info[m][idx] = data[7*m+1+idx][1:len(data[7*m+1+idx])]
            elif idx == 5:
                arm_info[m][idx] = data[7*m+1+idx][:-1]
            else:
                arm_info[m][idx] = data[7*m+1+idx]

regret_transfer = np.zeros((6,M))

# for print
lastpull = np.zeros((M,6,6)) #(M, nbPolicies, numArms)
lastmean = np.zeros((M,6,6))

##### Transfer Learning #####
Empirical_Lipschitz = np.zeros((6,M))   # store empirical Lipschitz constant every episode
# POLICIES = [{"archtype":OSSB_DEL, "params":{}}, {"archtype":LipschitzOSSB_DEL, "params":{}}]   
for m in range(M):
    POLICIES = []
    print("{} episode".format(m+1))
    # Generate model

    ENVIRONMENTS = [{"arm_type": Bernoulli, "params": arm_info[m]}]

    ##### Transfer Learning #####
    for idx in range(len(valuelist)):
        # OSSB_beta
        POLICIES.append(LipschitzOSSB_DEL_beta(nbArms=6, gamma=0.001, L=L_info[idx][m]))

    ##### true #####
    trueLC = com_Lipschitz_constant(arm_info[m], embeddings)
    POLICIES.append(LipschitzOSSB_DEL_true(nbArms=6, gamma=0.001, L=trueLC))

    ##### max #####
    POLICIES.append(LipschitzOSSB_DEL_beta(nbArms=6, gamma=0.001, L=L_info[3][m]))

    ##### inf #####
    POLICIES.append({"archtype":OSSB_DEL, "params":{}})

    configuration = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
    evaluation = Evaluator(configuration)
    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
        evaluation.startOneEnv(envId, env)
    for idx in range(len(POLICIES)):
        regret_transfer[idx][m] = evaluation.getCumulatedRegret_LessAccurate(policyId=idx)[HORIZON-1]
    for idx in range(len(POLICIES)):
        Empirical_Lipschitz[idx][m] = com_Lipschitz_constant(evaluation.get_lastmeans()[idx][0], embeddings)

    # self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)
    for idx in range(len(POLICIES)):
        lastpull[m][idx] = evaluation.getLastPulls()[idx, :, 0]
    for idx in range(len(POLICIES)):
        lastmean[m][idx] = evaluation.get_lastmeans()[idx][0]
    #evaluation.estimatedLipschitzdata(filepath=dir_name)

colors = ['tomato', 'limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple'] # 7
labels = ['(0.1, 0.05)', '(0.3, 0.05)', '(0.5, 0.05)', 'true', 'max', 'inf']
def plotRegret(filepath):
    plt.figure()
    X = list(range(1, M+1))
    for i in range(len(POLICIES)):
        plt.plot(X, regret_transfer[i], label="{}".format(labels[i]), color=colors[i])
    plt.legend()
    plt.title("Total {} episode, {} horizon".format(M, HORIZON))
    plt.savefig(filepath+'/Regret', dpi=300)

with open(dir_name+ "/EmpiricalLipschitz.txt", "w") as f:
    for i in range(len(Empirical_Lipschitz)):
        f.write("\n{}".format(i))
        for episode in range(len(Empirical_Lipschitz[i])):
            f.write("\nepisode:{}, {}".format(episode, Empirical_Lipschitz[i, episode]))

with open(dir_name+ "/regret_transfer.txt", "w") as f:
    for episode in range(M): 
        f.write("\nepisode:{}, {}, {}, {}, {}, {}, {}".format(episode, regret_transfer[0, episode], regret_transfer[1, episode], regret_transfer[2, episode], regret_transfer[3, episode], regret_transfer[4, episode], regret_transfer[5, episode]))


#lastpull = np.zeros((M,7,10)) #(M, nbPolicies, numArms)
with open(dir_name+ "/lastpull.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}".format(m))
        for policyId in range(6):
            f.write("\npolicy:{}\n".format(policyId))
            np.savetxt(f, lastpull[m][policyId].astype(int), fmt='%i', newline=", ")

with open(dir_name+ "/lastmean.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}".format(m))
        for policyId in range(6):
            f.write("\npolicy:{}\n".format(policyId))
            np.savetxt(f, lastmean[m][policyId], newline=", ", fmt='%1.3f')

with open(dir_name+ "/information.txt", "w") as f:
    f.write("Number of episodes: " + str(M))
    f.write("\nHorizon: " + str(HORIZON))
    f.write("\n(beta, epsilon): " + str(valuelist))
    f.write("\nembeddings: "+str(embeddings))

plotRegret(dir_name)
