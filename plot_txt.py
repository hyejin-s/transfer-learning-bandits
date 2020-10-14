import numpy as np
import matplotlib.pyplot as plt
import re
M = 100
smooth = 1
filepath = r"C:\Users\parke\bandit\checking"

with open(r"C:\Users\parke\bandit\checking\regret_transfer.txt") as f:
    data = f.read().split()
    plt.figure()
    X = np.linspace(0, 1, int(M/smooth))
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
    X = np.linspace(0, 1, int(M/smooth))
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



with open(r"C:\Users\parke\bandit\checking\L_beta.txt") as f:
    data = f.read().split()
    X = np.linspace(0, 1, int(M/smooth))
    plt.figure()
    for i in range(4):
        constant_list = []
        for m in range(M):
            constant_list.append(float(data[i*(M*2+1)+m*2+2]))
        Y_list = []
        for j in range(int(M/smooth)):
            Y_list.append(np.sum(constant_list[j*smooth:(j+1)*smooth]))
            
        plt.plot(X, Y_list, label="{}".format(i))
    plt.legend()
    plt.savefig(filepath+'/L_beta{}'.format(smooth), dpi=300)


with open(r"C:\Users\parke\bandit\checking\EmpiricalLipschitz.txt") as f:
    data = f.read().split()
    for i in range(7):
        plt.figure()
        constant_list = []
        for m in range(M):
            constant_list.append(float(data[i*(M*2+1)+m*2+2]))
            
        plt.hist(constant_list, label="{}".format(i+1))
        plt.legend()
        plt.savefig(filepath+'/histogram{}'.format(i+1), dpi=300)
