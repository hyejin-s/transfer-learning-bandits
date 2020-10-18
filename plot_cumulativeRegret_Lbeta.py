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

plt.savefig(filepath+'/cumregret', dpi=300)
plt.savefig(filepath+'/cumregret.pdf', format='pdf', dpi=300) 


"""
width = 0.4 # the width of the bars
for i in range(len(X)):
    X1.append(X[i]+0.2)
    X2.append(X[i]-0.2)
"""
# for comparing difference value_histogram
with open(r"C:\Users\parke\bandit\checking\regret_transfer.txt") as f:
    data = f.read().split()
    # for ture
    regret_true = []
    for m in range(M):
        regret_true.append(float(data[7*m+4][:-1]))

    regret_difference = np.zeros((6,M))
    for i in range(6):
        if i != 3:
            for m in range(M):
                if data[7*m+(i+1)][-1] == ',':
                    regret_difference[i][m] = (float(data[7*m+(i+1)][:-1])-regret_true[m])            
                else:
                    regret_difference[i][m] = (float(data[7*m+(i+1)])-regret_true[m])

    plt.figure()        
            # Make a separate list for each airline
    x1 = list(regret_difference[0])
    x2 = list(regret_difference[1])
    x3 = list(regret_difference[2])
    x4 = list(regret_difference[4])
    x5 = list(regret_difference[5])

    # Assign colors for each airline and the names
    colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
    labels2 = ['(0.1, 0.05)', '(0.3, 0.05)', '(0.5, 0.05)', 'max', 'inf']
         
    # Make the histogram using a list of lists
    # Normalize the flights and assign colors and names
    plt.hist([x1, x2, x3, x4, x5], bins = int(180/15), color = colors, label=labels2)

    # Plot formatting
    plt.legend()
    plt.show()    

#plt.hist(regret_difference, range=(-400,800), bins=10, rwidth=0.1, label=labels[i])

"""
plt.ylabel('Cumulative Regret', fontsize=60)
plt.xlabel('Episode', fontsize=60)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.legend(fontsize=40)

plt.savefig(filepath+'/cumregret', dpi=300)
plt.savefig(filepath+'/cumregret.pdf', format='pdf', dpi=300) 
"""