
import matplotlib.pyplot as plt
import numpy as np

filepath = r"C:\Users\parke\last_data"

from collections import defaultdict
with open(r"C:\Users\parke\last_data\estimated_Lipschitz_Constant_last.txt") as f:
    data = f.read().split()
# print(data)
Y = []
for i in range(100):
    Y.append(float(data[i*3+2]))
plt.figure(figsize=(20,15))
plt.hist(Y, bins=10, color='r', rwidth=0.9)
# plt.show()

# x=[10.5,70,92,194.5]
x = [10.5, 194.5]
y1=[16, 83.8]
y=[0.1010, 200.8184]
for i, v in enumerate(x):
    plt.text(v, y1[i], y[i],  
             fontsize = 40, 
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')

plt.xlabel('Estimated Lipschitz Constant', fontsize=60)
plt.ylabel('Number of Repetitions', fontsize=60)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.grid(b=None)


plt.savefig(filepath+'/LCHistogram_change', dpi=300)
plt.savefig(filepath+'/LCHistogram_change.pdf', format='pdf', dpi=300)


# Q = defaultdict(list)
# for i in range(100):
#     a = Y[i] // (float((max(Y)+1)/10))
#     Q[a].append(float(Y[i]))

# for i in Q.keys():
#     print(Q[i])
#     print(len(Q[i]))
#     Q[i] = np.mean(Q[i])
# print(Q)
