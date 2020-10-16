import numpy as np
import matplotlib.pyplot as plt
import re
M = 400
smooth = 1
valuelist = [[0.1, 0.05], [0.3, 0.05], [0.5, 0.05]]
labels = ['(0.1, 0.05)', '(0.3, 0.05)', '(0.5, 0.05)', 'true', 'max', 'inf']
filepath = r"C:\Users\parke\bandit\checking"

with open(r"C:\Users\parke\bandit\checking\regret_transfer.txt") as f:
    data = f.read().split()
    plt.figure(figsize=(24,16))
    X = list(range(1, int(M)+1))
    for i in range(6):
    
        regret_list = []
        for m in range(M):
            if data[7*m+(i+1)][-1] == ',':
                regret_list.append(float(data[7*m+(i+1)][:-1]))            
            else:
                regret_list.append(float(data[7*m+(i+1)]))
        plt.plot(X, np.cumsum(regret_list), label="{}".format(labels[i]),linewidth=5)


plt.ylabel('Cumulative Regret', fontsize=60)
plt.xlabel('Episode', fontsize=60)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.legend(fontsize=40)

plt.savefig(filepath+'/cumregret1', dpi=300)
plt.savefig(filepath+'/cumregret1.pdf', format='pdf', dpi=300) 