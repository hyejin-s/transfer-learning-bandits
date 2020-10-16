import numpy as np
import matplotlib.pyplot as plt
import re
from math import ceil
M = 400
smooth = 1
#[beta, epsilon], [0, 0] for max
valuelist = [[0, 0], [0.5, 0.05], [0.3, 0.05], [0.1, 0.05]]
filepath = r"C:\Users\parke\bandit\checking"

def Lipschitz_beta(L_list, epsilon, beta, M_present):
    bound_value = ceil(beta*M_present) # ROUND UP
    #L_list.sort(reverse=True)
    L_beta = sorted(L_list, reverse=True)[bound_value-1] + epsilon
    return L_beta

arm_info = np.zeros((M,6))
with open(r"C:\Users\parke\bandit\checking\arm_info.txt") as f:
    data = f.read().split()
    for m in range(M):
        for idx in range(6):
            if idx == 0:
                arm_info[m][idx] = data[7*m+1+idx][1:len(data[7*m+1+idx])]
            elif idx == 5:
                arm_info[m][idx] = data[7*m+1+idx][:-1]
            else:
                arm_info[m][idx] = data[7*m+1+idx]

def com_Lipschitz_constant(thetas, embeddings):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i+1]-thetas[i])/(embeddings[i+1]-embeddings[i]))
    return np.amax(L_values)

embeddings = [0, 0.8, 0.85, 0.9, 0.95, 1]
Y = []
for m in range(M):
    Y.append(com_Lipschitz_constant(arm_info[m],embeddings))

with open(r"C:\Users\parke\bandit\checking\EmpiricalLipschitz.txt") as f:
    data = f.read().split()
    plt.figure(figsize=(24,16))
    X = list(range(1,M+1))

    for i in range(len(valuelist)):
        Lip_list = []
        Empirical_list = []
        if i == 0:
            for m in range(M):
                Empirical_list.append(float(data[m*2+1]))
                Lip_list.append(max(Empirical_list))
            plt.plot(X, Lip_list, label=r"max($\hat{L}_m$)", linewidth=5)
        else:
            beta = valuelist[i][0]
            epsilon = valuelist[i][1]
            for m in range(M):
                Empirical_list.append(float(data[m*2+1]))
                Lipb = Lipschitz_beta(Empirical_list, epsilon, beta, m)
                Lip_list.append(Lipb)
            plt.plot(X, Lip_list, label=r"$(\beta, \epsilon)$=({}, {})".format(valuelist[i][0], valuelist[i][1]), linewidth=5)

    plt.plot(X, Y, label="True Lipschitz Constant", linewidth=5)
    plt.legend(loc='upper right')
    plt.ylabel('Lipschitz Constant', fontsize=60)
    plt.xlabel('Episode', fontsize=60)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    plt.legend(fontsize=40, loc="upper right")

    plt.savefig(filepath+'/Lbeta', dpi=300)
    plt.savefig(filepath+'/Lbeta.pdf', format='pdf', dpi=300) 

# true
plt.figure(figsize=(20,15))
plt.hist(Y, rwidth=0.9, color='r')
plt.ylabel('Number of Episodes', fontsize=60)
plt.xlabel('Lipschitz Constant', fontsize=60)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.savefig(filepath+'/trueLC_hist', dpi=300)
plt.savefig(filepath+'/trueLC_hist.pdf', format='pdf', dpi=300) 

# empirical
plt.figure(figsize=(20,15))
plt.hist(Empirical_list, rwidth=0.9, color='b')
plt.ylabel('Number of Episodes', fontsize=60)
plt.xlabel(r'$\hat{L}_m$', fontsize=60)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.savefig(filepath+'/empiLC_hist', dpi=300)
plt.savefig(filepath+'/empiLC_hist.pdf', format='pdf', dpi=300) 