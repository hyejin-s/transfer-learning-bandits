import numpy as np
import matplotlib.pyplot as plt
import re
from math import log, ceil

M = 400
smooth = 1
data_name = ['inf', 'true', 'estimate']
valuelist =  [[0.5, 0.1], [0.2, 0.1]]
filepath = r"C:\Users\parke\bandit\checking"

colors = ['tomato','limegreen', 'dodgerblue', 'gold']

def Lipschitz_beta(L_list, epsilon, beta, M_present):
    bound_value = ceil(beta*M_present) # ROUND UP
    #L_list.sort(reverse=True)
    L_beta = sorted(L_list, reverse=True)[bound_value-1] + epsilon*beta
    return L_beta

beta_list = np.zeros((2, M))
max_list = np.zeros((2, M))
with open(r"C:\Users\parke\bandit\checking\EmpiricalLipschitz.txt") as f:
    data = f.read().split()
    for i in range(2):
        Empirical_list = []
        A = []
        for m in range(101):
            if i ==0:
                Empirical_list.append(float(data[(2*i)*(M*2+1)+m*2+2]))
                Lipb = Lipschitz_beta(Empirical_list, 0.1, 0.5, m)
                beta_list[i][m] = Lipb
            else:
                Empirical_list.append(float(data[(2*i)*(M*2+1)+m*2+2]))
                Lipb = Lipschitz_beta(Empirical_list, 0.1, 0.2, m)
                beta_list[i][m] = Lipb
        for m in range(101):
            A.append(float(data[(2*i+1)*(M*2+1)+m*2+2]))
            max_list[i][m] = max(A)
            
# compare graph
plt.figure()
X = list(range(1,int(M/smooth)+1))
with open(r"C:\Users\parke\bandit\checking\L_beta.txt") as f:
    data = f.read().split()
    for i in range(2):
        for m in range(M):
            if m > 100:
                beta_list[i][m] = float(data[(i)*(M*2+1)+m*2+2])
        Y_list = []
        for j in range(int(M/smooth)):
            Y_list.append(np.sum(beta_list[i][j*smooth:(j+1)*smooth])/smooth)
        plt.plot(X, Y_list, color=colors[i], label="{}".format(valuelist[i]))

with open(r"C:\Users\parke\bandit\checking\trueLC.txt") as f:
    data = f.read().split()
    X = list(range(1,int(M/smooth)+1))
    true_list = []
    for m in range(M):
        true_list.append(float(data[m*2+1])) 
    Y_list = []
    for j in range(int(M/smooth)):
        Y_list.append(np.sum(true_list[j*smooth:(j+1)*smooth])/smooth)

plt.plot(X, Y_list, color=colors[3], label="trueLC")

with open(r"C:\Users\parke\bandit\checking\L_max.txt") as f:
    data = f.read().split()
    for i in range(2):
        for m in range(M):
            if m > 100:
                max_list[i][m] = float(data[(i)*(M*2+1)+m*2+2])
        Y_list = []
        for j in range(int(M/smooth)):
            Y_list.append(np.sum(max_list[i][j*smooth:(j+1)*smooth])/smooth)
        plt.plot(X, Y_list, label="{}".format(valuelist[i]))
plt.legend()
plt.savefig(filepath+'/L_compare_whole{}'.format(smooth), dpi=300)



with open(r"C:\Users\parke\bandit\checking\regret_transfer.txt") as f:
    data = f.read().split()
    plt.figure()
    X = list(range(1, int(M/smooth)+1))
    colors = ['tomato', 'orangered','limegreen', 'dodgerblue']
    for i in range(4):
        regret_list = []
        for m in range(M):
            if data[5*m+(i+1)][-1] == ",":
                regret_list.append(float(data[5*m+(i+1)][0:-2]))
            else:
                regret_list.append(float(data[5*m+(i+1)][0:-1]))
        Y_list = []
        for j in range(int(M/smooth)):
            Y_list.append(np.sum(regret_list[j*smooth:(j+1)*smooth])/smooth)
            
        plt.plot(X, Y_list, color=colors[i], label="{}".format(i))
    plt.legend()


with open(r"C:\Users\parke\bandit\checking\regret_info.txt") as f:
    data = f.read().split()
    plt.figure()
    X = list(range(1, int(M/smooth)+1))
    colors = ['tomato','limegreen', 'dodgerblue']
    for i in range(3):
        regret_list = []
        for m in range(M):
            if data[4*m+(i+1)][-1] == ",":
                regret_list.append(float(data[4*m+(i+1)][0:-2]))
            else:
                regret_list.append(float(data[4*m+(i+1)][0:-1]))
        Y_list = []
        for j in range(int(M/smooth)):
            Y_list.append(np.sum(regret_list[j*smooth:(j+1)*smooth])/smooth)
            
        plt.plot(X, Y_list, color=colors[i], label="{}".format(i))
    plt.legend()