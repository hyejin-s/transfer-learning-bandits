import matplotlib.pyplot as plt
import numpy as np
import random

from OSSB_KL import solve_optimization_problem__classic, solve_optimization_problem__Lipschitz

num_arm = 6
embeddings = [0]
for i in range(num_arm-1):
    embeddings.append(0.8+ i*0.2 /(num_arm-2))

M = 400
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

plt.figure()
X = list(range(1,M+1))
Y = []
for m in range(M):
    arms = arm_info[m]
    zeta = np.zeros(6)
    classic = solve_optimization_problem__classic(arms, zeta)
    lip = solve_optimization_problem__Lipschitz(arms, zeta, L=0.5)
    Y.append(classic-lip)
    print(str(m) + " classic: " + str(classic) + ", lip: " + str(lip) +"-------- " +str(classic-lip))
plt.plot(X, np.cumsum(Y), linewidth=1)
plt.show()


#filepath = r"C:\\Users\\parke\\OneDrive\\바탕 화면\\fig"
#plt.savefig(filepath+'/gene', dpi=300)
#plt.savefig(filepath+'/gene.pdf', format='pdf', dpi=300) 