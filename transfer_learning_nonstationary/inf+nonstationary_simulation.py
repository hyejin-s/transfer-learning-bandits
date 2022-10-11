name = ['/Dataset 0.csv','/Dataset 1.csv','/Dataset 2.csv','/Dataset 3.csv',
        '/Dataset 4.csv','/Dataset 5.csv','/Dataset 6.csv','/Dataset 7.csv']


from Bernolli import Bernoulli
from evaluator_transferlearning import *
from OSSB_Transfer10 import OSSB_DEL, LipschitzOSSB_DEL, LipschitzOSSB_DEL_beta, LipschitzOSSB_DEL_true

import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log, ceil, sqrt
import os
import numpy as np
from scipy import interpolate 

HORIZON=10000
REPETITIONS=1
N_JOBS=40

sliding_window=3
num = 293
M = num - sliding_window + 1 # the ordering of Transfer Learning

######################################### def #########################################

def Lipschitz_beta(L_list, epsilon, beta, M_present):
    bound_value = ceil(beta*M_present) # ROUND UP
    # L_list.sort(reverse=True)
    L_beta = sorted(L_list, reverse=True)[bound_value-1] + epsilon
    return L_beta

def com_Lipschitz_constant(thetas, embeddings):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs((thetas[i+1]-thetas[i])/(embeddings[i+1]-embeddings[i])))
        
    return np.amax(L_values)
    
########################################################################################

# arm inf
value_save = np.zeros((8,num))
for i in range(8):
    data = np.loadtxt(r'/home/phj/bandits/data'+name[i], delimiter=',', dtype = np.float32) 
    x_ori = data[:,0] 
    y_ori = data[:,1] 
    f1 = interpolate.interp1d(x_ori,y_ori) 
    x_new = np.linspace(3,296,num=num,endpoint=True)
    value_save[i]=f1(x_new)
    undervalue = np.where(value_save[i] < 0.05)[0]
    for j in range(len(undervalue)):
        value_save[i][undervalue[j]] = 0.05

embeddings = [54/54, 48/54, 36/54, 24/54, 18/54, 12/54, 9/54, 6/54]
for i in range(8):
    value_save[i] = value_save[i]/(embeddings[i]*54)   # success rate

dir_name = "./non-stationary"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

valuelist = [[0.1, 0.1], [0.3, 0.1], [0.5, 0.1]] # [beta, epsilon]
Lbeta_info = np.zeros((len(valuelist), M))
Ltrue_info = np.zeros(M)
nbPolicies = 5
numArms = 8


arm_info = np.zeros((M, len(embeddings)))
for i in range(len(embeddings)):
    for m in range(M):
        arm_info[m][i] = np.sum(value_save[i][m:m+sliding_window])/sliding_window

with open(dir_name+ "/arm_info.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{} ".format(m))
        np.savetxt(f, arm_info[m], newline=", ", fmt='%1.3f')


# plt.figure()
# plt.ylim(0, 1)
# for i in range(10):
#     plt.plot(embeddings, arm_info[i], label="{}".format(i))
# plt.legend()
# plt.show()


#################### numpy for saving info ####################
regret_transfer = np.zeros((nbPolicies, M)) 
Empirical_Lipschitz = np.zeros((nbPolicies, M))  # store empirical Lipschitz constant every episode
defalut_Lipschitz = np.zeros(num)

# for print
lastpull = np.zeros((M, nbPolicies, numArms)) #(M, nbPolicies, numArms)
lastmean = np.zeros((M, nbPolicies, numArms))

#################### \pi(\inf) info save ####################
# POLICIES[0] = \inf
for m in range(M):
    print("{} episode".format(m+1))

    ENVIRONMENTS = [{"arm_type": Bernoulli, "params": arm_info[m]}]
    POLICIES = [{"archtype":OSSB_DEL, "params":{}}]

    configuration = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
    evaluation = Evaluator(configuration)
    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
        evaluation.startOneEnv(envId, env)

    # POLICIES[0] = \inf
    regret_transfer[0][m] = evaluation.getCumulatedRegret_LessAccurate(policyId=0)[HORIZON-1]
    defalut_Lipschitz[m] = com_Lipschitz_constant(evaluation.get_lastmeans()[0][0], embeddings)
    # self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)
    lastpull[0][0] = evaluation.getLastPulls()[0, :, 0]
    lastmean[0][0] = evaluation.get_lastmeans()[0][0]
    #evaluation.estimatedLipschitzdata(filepath=dir_name)


with open(dir_name+ "/EmpiricalLipschitz.txt", "w") as f:
    for episode in range(num):
        f.write("\nepisode:{}, {}".format(episode, defalut_Lipschitz[episode]))

#### for sliding_window
for m in range(M):
    Empirical_Lipschitz[0][m] = np.sum(defalut_Lipschitz[m:m+sliding_window])/sliding_window

#### for \beta Lipschitz constant information
for i in range(len(valuelist)):
    Empirical_list = []
    beta = valuelist[i][0]
    epsilon = valuelist[i][1]
    for m in range(M):
        Empirical_list.append(Empirical_Lipschitz[0][m])
        Lbeta_info[i][m] = Lipschitz_beta(Empirical_list, epsilon, beta, m)

# POLICIES = [{"archtype":OSSB_DEL, "params":{}}, {"archtype":LipschitzOSSB_DEL, "params":{}}]   
for m in range(M):
    POLICIES = []
    print("{} episode".format(m+1))
    # Generate model

    ENVIRONMENTS = [{"arm_type": Bernoulli, "params": arm_info[m]}]

    ##### Transfer Learning #####
    for idx in range(len(valuelist)):
        # OSSB_beta
        POLICIES.append(LipschitzOSSB_DEL_beta(nbArms=8, gamma=0.001, L=Lbeta_info[idx][m]))
        print(Lbeta_info)
    ##### true #####
    trueLC = com_Lipschitz_constant(arm_info[m], embeddings)
    Ltrue_info[m] = trueLC
    POLICIES.append(LipschitzOSSB_DEL_true(nbArms=8, gamma=0.001, L=trueLC))

    # ##### max #####
    # POLICIES.append(LipschitzOSSB_DEL_beta(nbArms=8, gamma=0.001, L=Lbeta_info[3][m]))
    # ##### inf #####
    # POLICIES.append({"archtype":OSSB_DEL, "params":{}})

    configuration = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
    evaluation = Evaluator(configuration)
    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
        evaluation.startOneEnv(envId, env)
    for idx in range(len(POLICIES)):
        regret_transfer[idx+1][m] = evaluation.getCumulatedRegret_LessAccurate(policyId=idx)[HORIZON-1]
    for idx in range(len(POLICIES)):
        Empirical_Lipschitz[idx+1][m] = com_Lipschitz_constant(evaluation.get_lastmeans()[idx][0], embeddings)

    # self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)
    for idx in range(len(POLICIES)):
        lastpull[m][idx+1] = evaluation.getLastPulls()[idx, :, 0]
    for idx in range(len(POLICIES)):
        lastmean[m][idx+1] = evaluation.get_lastmeans()[idx][0]
    #evaluation.estimatedLipschitzdata(filepath=dir_name)

with open(dir_name+ "/Lipschitz_info.txt", "w") as f:
    # for m in range(M):
    #     f.write("\nepisode:{} ".format(m))
    np.savetxt(f, Empirical_Lipschitz[0], newline=", ", fmt='%1.3f')

colors = ['tomato', 'limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple'] # 7
labels = ['inf', '(0.1, 0.1)', '(0.3, 0.1)', '(0.5, 0.1)', 'true']

def plotRegret(filepath):
    plt.figure()
    X = list(range(1, M+1))
    for i in range(len(POLICIES)):
        plt.plot(X, regret_transfer[i], label="{}".format(labels[i]), color=colors[i])
    plt.legend()
    plt.title("Total {} episode, {} horizon".format(M, HORIZON))
    plt.savefig(filepath+'/Regret', dpi=300)
    plt.savefig(filepath+'/Regret.pdf', format='pdf', dpi=300) 

def plotLbeta(filepath):
    plt.figure()
    X = list(range(1, M+1))
    for i in range(len(POLICIES)+1):
        if i == 0: # inf
            # plt.plot(X, Empirical_Lipschitz[0], label="{}".format(labels[i]), color=colors[i])
            pass
        elif i == len(POLICIES): # true
            plt.plot(X, Ltrue_info, label="{}".format(labels[i]), color=colors[i])
        else:
            plt.plot(X, Lbeta_info[i-1], label="{}".format(labels[i]), color=colors[i])
    plt.legend()
    plt.title("Total {} episode, {} horizon".format(M, HORIZON))
    plt.savefig(filepath+'/Lbeta', dpi=300)
    plt.savefig(filepath+'/Lbeta.pdf', format='pdf', dpi=300) 

with open(dir_name+ "/EmpiricalLipschitz.txt", "w") as f:
    for i in range(len(Empirical_Lipschitz)):
        f.write("\n{}".format(i))
        for episode in range(len(Empirical_Lipschitz[i])):
            f.write("\nepisode:{}, {}".format(episode, Empirical_Lipschitz[i, episode]))

with open(dir_name+ "/regret_transfer.txt", "w") as f:
    for episode in range(M): 
        f.write("\nepisode:{}, {}, {}, {}, {}, {}".format(episode, regret_transfer[0, episode], regret_transfer[1, episode], regret_transfer[2, episode], regret_transfer[3, episode], regret_transfer[4, episode]))

#lastpull = np.zeros((M,7,10)) #(M, nbPolicies, numArms)
with open(dir_name+ "/lastpull.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}".format(m))
        for policyId in range(5):
            f.write("\npolicy:{}\n".format(policyId))
            np.savetxt(f, lastpull[m][policyId].astype(int), fmt='%i', newline=", ")

with open(dir_name+ "/lastmean.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}".format(m))
        for policyId in range(5):
            f.write("\npolicy:{}\n".format(policyId))
            np.savetxt(f, lastmean[m][policyId], newline=", ", fmt='%1.3f')

with open(dir_name+ "/information.txt", "w") as f:
    f.write("Number of episodes: " + str(M))
    f.write("\nHorizon: " + str(HORIZON))
    f.write("\n(beta, epsilon): " + str(valuelist))
    f.write("\nembeddings: "+str(embeddings))

plotRegret(dir_name)
plotLbeta(dir_name)