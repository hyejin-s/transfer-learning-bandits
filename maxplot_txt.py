import numpy as np
import matplotlib.pyplot as plt
import re
M = 400
smooth = 1
data_name = ['inf', 'true', 'estimate']
valuelist = [[0.2, 0.2], [0.5, 0.1], [0.2, 0.1], [0.25, 0.1]]
filepath = r"C:\Users\parke\bandit\checking"

colors = ['tomato','limegreen', 'dodgerblue', 'gold']


# compare graph
plt.figure()
X = list(range(1,int(M/smooth)+1))
with open(r"C:\Users\parke\bandit\checking\L_beta.txt") as f:
    data = f.read().split()
    for i in range(2):
        beta_list = []
        for m in range(M):
            beta_list.append(float(data[(i)*(M*2+1)+m*2+2])) 
        Y_list = []
        for j in range(int(M/smooth)):
            Y_list.append(np.sum(beta_list[j*smooth:(j+1)*smooth])/smooth)
        plt.plot(X, Y_list, color=colors[i], label="{}".format(i))


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
        max_list = []
        for m in range(M):
            max_list.append(float(data[(i)*(M*2+1)+m*2+2])) 
        Y_list = []
        for j in range(int(M/smooth)):
            Y_list.append(np.sum(max_list[j*smooth:(j+1)*smooth])/smooth)
        plt.plot(X, Y_list, label="{}".format(i))
plt.legend()
plt.savefig(filepath+'/L_compare{}'.format(smooth), dpi=300)

