
import matplotlib.pyplot as plt
import numpy as np


filepath = r"C:\Users\parke\last_data"

from collections import defaultdict
with open(r"C:\Users\parke\last_data\estimated_Lipschitz_Constant.txt") as f:
    data = f.read().split()

min_list = [0, 31, 34, 46, 48, 50, 53, 63, 85, 95]  # brought from result of broken_axis.py
max_list = [6, 10, 21, 39, 42, 47, 59, 64, 88, 94]

# min_list=[0, 14, 16, 20, 31, 34, 41, 43, 46, 48, 50, 53, 56, 63, 72, 79, 85, 87, 95, 99]
# max_list=[1, 6, 10, 12, 17, 21, 22, 29, 42, 45, 47, 59, 64, 67, 69, 83, 88, 94, 97, 98]

bottom_timelist = []
top_timelist = []
list_200 = []
for j in range(50000):
    a, b = 0, 0
    for i in range(100):
        if int(data[(50000*2+2)*i+1]) in min_list:
            # print(data[(50000*2+2)*i+1])
            # print(data[(50000*2+2)*i + 1 + 2*(j+1)])
            if not data[(50000*2+2)*i+1+2*(j+1)] == "nan" or "0.0":    
                a += float(data[(50000*2+2)*i + 1 + 2*(j+1)])
        if int(data[(50000*2+2)*i+1]) in max_list:
            # print(data[(50000*2+2)*i+1])
            # print(data[(50000*2+2)*i + 1 + 2*(j+1)])
            if not data[(50000*2+2)*i+1+2*(j+1)] == "nan" or "0.0":    
                b += float(data[(50000*2+2)*i + 1 + 2*(j+1)])        
    bottom_timelist.append(a/10)
    top_timelist.append(b/10)
    list_200.append(200)

colors = ['red', 'blue', 'green']
plt.figure(figsize=(20,15))
X = list(range(0, 50000))
plt.plot(X, bottom_timelist, linewidth = 5, color=colors[0], label="bottom 10.0%")
plt.plot(X, top_timelist, linewidth = 5, color=colors[1], label="top 10.0%")
plt.plot(X, list_200, linewidth = 5, color=colors[2], label="True Lipschitz Constant")

plt.legend(fontsize=40)

plt.xlabel('Time', fontsize=60)
plt.ylabel('Lipschitz Constant', fontsize=60)
plt.xticks([0,10000,20000,30000,40000,50000], ['0', '10k', '20k', '30k', '40k', '50k'])
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

plt.savefig(filepath+'/Estimated_L', dpi=300)
plt.savefig(filepath+'/Estimated_L.pdf', format='pdf', dpi=300)


# fig, axs = plt.subplots(2)
# fig.suptitle('Vertically stacked subplots')
# axs[0].plot(X, bottom_timelist)
# axs[1].plot(X, top_timelist)

# plt.figure(figsize=(20,15))
# X = list(range(0, 50000))
# plt.plot(X, bottom_timelist, linewidth = 5, color=colors[0], label="bottom 10.0%")
# plt.plot(X, top_timelist, linewidth = 5, color=colors[1], label="top 10.0%")
# plt.plot(X, list_200, linewidth = 5, color=colors[2], label="True Lipschitz Constant")

# plt.legend(fontsize=40)

# plt.xlabel('Time', fontsize=60)
# plt.ylabel('Lipschitz Constant', fontsize=60)
# plt.xticks(fontsize=40)
# plt.yticks(fontsize=40)

# plt.savefig(filepath+'/Estimated_L_subplot', dpi=300)
# plt.savefig(filepath+'/Estimated_L_subplot.pdf', format='pdf', dpi=300)


