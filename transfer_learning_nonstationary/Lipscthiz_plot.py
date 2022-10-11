import numpy as np
import matplotlib.pyplot as plt
import re
from math import ceil


# Lipschitz_info = []
# with open(r"/home/phj/bandits/non-stationary-2/Lipschitz_info.txt") as f:
#     data = f.read().split()
#     # print(data[0][0:5])
#     for i in range(len(data)):
#         Lipschitz_info.append(float(data[i][0:5]))
# # print(Lipschitz_info)

plt.figure(figsize=(24,16))
X = list(range(1,len(data)+1))
plt.plot(X, Lipschitz_info, label="Lipschitz Constant", linewidth=5)
plt.legend(loc='upper right')
plt.ylabel('Lipschitz Constant', fontsize=60)
plt.xlabel('Episode', fontsize=60)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

plt.legend(fontsize=40, loc="upper right")

plt.savefig('./Lipschitz_ture', dpi=300)
plt.savefig('./Lipschitz_ture.pdf', format='pdf', dpi=300) 

