name = ['/Dataset 0.csv','/Dataset 1.csv','/Dataset 2.csv','/Dataset 3.csv',
        '/Dataset 4.csv','/Dataset 5.csv','/Dataset 6.csv','/Dataset 7.csv']

from Bernolli import Bernoulli
from evaluator_transferlearning import *
from OSSB_Transfer10_d import OSSB_DEL, distanceOSSB_DEL_beta, distanceOSSB_DEL_true 

import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log, ceil
import os
import numpy as np
from scipy import interpolate 

HORIZON=5000
REPETITIONS=1
N_JOBS=40

sliding_window = 5
num = 293
M = num - sliding_window + 1 # the ordering of Transfer Learning

# arm info
value_save = np.zeros((8,num))
for i in range(8):
    data = np.loadtxt(r'/home/phj/bandits/data'+name[i], delimiter=',', dtype = np.float32)     
    x_ori = data[:,0] 
    y_ori = data[:,1] 
    f1 = interpolate.interp1d(x_ori,y_ori) 
    x_new = np.linspace(3,296,num=num,endpoint=True)
    value_save[i]=f1(x_new)
    undervalue = np.where(value_save[i] < 0)[0]
    for j in range(len(undervalue)):
        value_save[i][undervalue[j]] = 0.01

# embeddings = [54/54, 48/54, 36/54, 24/54, 18/54, 12/54, 9/54, 6/54]
rates = [54, 48, 36, 24, 18, 12, 9, 6]

# success rate
for i in range(8):
    value_save[i] = value_save[i]/rates[i] 

dir_name = "./non-stationary-d"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

valuelist = [[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.1, 0.05], [0.3, 0.05], [0.5, 0.05], [0.1, 0.5], [0.3, 0.5], [0.5, 0.5]] # [beta, epsilon]
dbeta_info = np.zeros((len(valuelist), M, 8, 8))
dtrue_info = np.zeros((M, 8, 8))

nbPolicies = 11
numArms = 8

arm_info = np.zeros((M, len(rates)))
for i in range(len(rates)):
    for m in range(M):
        arm_info[m][i] = np.sum(value_save[i][m:m+sliding_window])/sliding_window

with open(dir_name+ "/arm_info.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{} ".format(m))
        np.savetxt(f, arm_info[m], newline=", ", fmt='%1.3f')
        
with open(dir_name+ "/reward_info.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{} ".format(m))
        for i in range(numArms):
            f.write("{}, ".format(arm_info[m][i]*rates[i]))
        f.write("\\")


######################################### def #########################################

def distance_beta(d_list, epsilon, beta, M_present):
    bound_value = ceil(beta*M_present) # ROUND UP
    #L_list.sort(reverse=True)
    d_beta = sorted(d_list, reverse=True)[bound_value-1] + epsilon
    return d_beta

########################################################################################

#################### numpy for saving info ####################
regret_transfer = np.zeros((nbPolicies, M)) 
Empirical_distance = np.zeros((nbPolicies, M, 8, 8))  # store empirical Lipschitz constant every episode
defalut_distance = np.zeros((num,8,8))

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
    regret_transfer[0][m] = evaluation.getRewardCumulatedRegret_LessAccurate(policyId=0)[HORIZON-1]
    for i in range(8):
        for j in range(8):
            defalut_distance[m][i][j] = abs(evaluation.get_lastmeans()[0][0][i]-evaluation.get_lastmeans()[0][0][j]) 
    # self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)
    lastpull[m][0] = evaluation.getLastPulls()[0, :, 0]
    lastmean[m][0] = evaluation.get_lastmeans()[0][0]
    #evaluation.estimatedLipschitzdata(filepath=dir_name)

# for m in range(M):
#     for i in range(8):
#         for j in range(8):
#             Empirical_distance[0][m][i][j] = np.sum(defalut_distance[m:m+sliding_window][:,i][:,j])/sliding_window

#### for \beta information
for k in range(len(valuelist)):
    beta = valuelist[k][0]
    epsilon = valuelist[k][1]

    for i in range(8):
        for j in range(8):
            Empirical_list = []
            for m in range(M):
                Empirical_list.append(defalut_distance[m][i][j])
                dbeta_info[k][m][i][j] = distance_beta(Empirical_list, epsilon, beta, m)

# POLICIES = [{"archtype":OSSB_DEL, "params":{}}, {"archtype":LipschitzOSSB_DEL, "params":{}}]   
for m in range(M):
    POLICIES = []
    print("{} episode".format(m+1))
    # Generate model

    ENVIRONMENTS = [{"arm_type": Bernoulli, "params": arm_info[m]}]

    ##### Transfer Learning #####
    for idx in range(len(valuelist)):
        # OSSB_beta
        print(dbeta_info[idx][m])
        POLICIES.append(distanceOSSB_DEL_beta(nbArms=8, gamma=0.001, d=dbeta_info[idx][m]))

    ##### true #####
    true_d = np.zeros((8,8))
    for i in range(8):
        for j in range(8):
            true_d[i][j] = abs(arm_info[m][i]-arm_info[m][j])
    POLICIES.append(distanceOSSB_DEL_true(nbArms=8, gamma=0.001, d=true_d))

    # ##### max #####
    # POLICIES.append(LipschitzOSSB_DEL_beta(nbArms=8, gamma=0.001, L=Lbeta_info[3][m]))
    # ##### inf #####
    # POLICIES.append({"archtype":OSSB_DEL, "params":{}})

    configuration = {"horizon": HORIZON, "repetitions": REPETITIONS, "n_jobs": N_JOBS, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
    evaluation = Evaluator(configuration)
    for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
        evaluation.startOneEnv(envId, env)
    for idx in range(len(POLICIES)):
        regret_transfer[idx+1][m] = evaluation.getRewardCumulatedRegret_LessAccurate(policyId=idx)[HORIZON-1]
    # for idx in range(len(POLICIES)):
    #     # mu
    #     # mu = np.zeros(numArms)
    #     # for i in range(numArms):
    #     #     mu[i] = evaluation.get_lastmeans()[idx][0][i]*embeddings[i]
    #     # Empirical_Lipschitz[idx][m] = com_Lipschitz_constant(mu, embeddings)
    #     # theta
    #     Empirical_Lipschitz[idx][m] = com_Lipschitz_constant(evaluation.get_lastmeans()[idx][0], embeddings)

    # self.lastPulls[envId] = np.zeros((self.nbPolicies, self.envs[envId].nbArms, self.repetitions), dtype=np.int32)
    for idx in range(len(POLICIES)):
        lastpull[m][idx+1] = evaluation.getLastPulls()[idx, :, 0]
    for idx in range(len(POLICIES)):
        lastmean[m][idx+1] = evaluation.get_lastmeans()[idx][0]
    #evaluation.estimatedLipschitzdata(filepath=dir_name)

colors = ['tomato', 'limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple', 'navajowhite', 'lavender', 'darkgrey', 'royalblue'] # 11
labels = ['inf', '(0.1, 0.1)', '(0.3, 0.1)', '(0.5, 0.1)', '(0.1, 0.05)', '(0.3, 0.05)', '(0.5, 0.05)', '(0.1, 0.5)', '(0.3, 0.5)', '(0.5, 0.5)', 'true']
def plotRegret(filepath):
    plt.figure()
    X = list(range(1, M+1))
    for i in range(len(POLICIES)+1):
        plt.plot(X, regret_transfer[i], label="{}".format(labels[i]), color=colors[i])
    plt.legend()
    plt.title("Total {} episode, {} horizon".format(M, HORIZON))
    plt.savefig(filepath+'/Regret', dpi=300)
    plt.savefig(filepath+'/Regret.pdf', format='pdf', dpi=300) 

# with open(dir_name+ "/EmpiricalLipschitz.txt", "w") as f:
#     for i in range(len(Empirical_Lipschitz)):
#         f.write("\n{}".format(i))
#         for episode in range(len(Empirical_Lipschitz[i])):
#             f.write("\nepisode:{}, {}".format(episode, Empirical_Lipschitz[i, episode]))

with open(dir_name+ "/regret_transfer.txt", "w") as f:
    for episode in range(M): 
        f.write("\nepisode:{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}".format(episode, regret_transfer[0, episode], regret_transfer[1, episode], 
        regret_transfer[2, episode], regret_transfer[3, episode], regret_transfer[4, episode], regret_transfer[5, episode], regret_transfer[6, episode]
        , regret_transfer[7, episode], regret_transfer[8, episode], regret_transfer[9, episode], regret_transfer[10, episode]))

#lastpull = np.zeros((M,7,10)) #(M, nbPolicies, numArms)
with open(dir_name+ "/lastpull.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}".format(m))
        for policyId in range(nbPolicies):
            f.write("\npolicy:{}\n".format(policyId))
            np.savetxt(f, lastpull[m][policyId].astype(int), fmt='%i', newline=", ")

with open(dir_name+ "/lastmean.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}".format(m))
        for policyId in range(nbPolicies):
            f.write("\npolicy:{}\n".format(policyId))
            np.savetxt(f, lastmean[m][policyId], newline=", ", fmt='%1.3f')

with open(dir_name+ "/information.txt", "w") as f:
    f.write("Number of episodes: " + str(M))
    f.write("\nHorizon: " + str(HORIZON))
    f.write("\n(beta, epsilon): " + str(valuelist))
    f.write("\nembeddings: "+str(embeddings))


# def plotLbeta(filepath):
#     plt.figure()
#     X = list(range(1, M+1))
#     for i in range(len(POLICIES)+1):
#         if i == 0: # inf
#             # plt.plot(X, Empirical_Lipschitz[0], label="{}".format(labels[i]), color=colors[i])
#             pass
#         elif i == len(POLICIES): # true
#             plt.plot(X, Ltrue_info, label="{}".format(labels[i]), color=colors[i])
#         else:
#             plt.plot(X, Lbeta_info[i-1], label="{}".format(labels[i]), color=colors[i])
#     plt.legend()
#     plt.title("Total {} episode, {} horizon".format(M, HORIZON))
#     plt.savefig(filepath+'/Lbeta', dpi=300)
#     plt.savefig(filepath+'/Lbeta.pdf', format='pdf', dpi=300) 

plotRegret(dir_name)
# plotLbeta(dir_name)