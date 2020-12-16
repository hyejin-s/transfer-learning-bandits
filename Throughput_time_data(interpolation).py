name = ['\Dataset 0.csv','\Dataset 1.csv','\Dataset 2.csv','\Dataset 3.csv',
        '\Dataset 4.csv','\Dataset 5.csv','\Dataset 6.csv','\Dataset 7.csv']

import numpy as np 
import matplotlib.pyplot as plt 
from scipy import interpolate 
value_save = np.zeros((8,293))
for i in range(8):
    data = np.loadtxt(r'C:\Users\parke\OneDrive\바탕 화면\OptimalRateSampling_graph'+name[i], delimiter=',', dtype = np.float32) 
    x_ori = data[:,0] 
    y_ori = data[:,1] 
    f1 = interpolate.interp1d(x_ori,y_ori) 
    x_new = np.linspace(3,296,num=293,endpoint=True)
    value_save[i]=f1(x_new)


from math import log
embeddings = [54, 48, 36, 24, 18, 12, 9, 6]

def estimate_Lipschitz_constant(thetas):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i]-thetas[i+1])/abs(1/embeddings[i+1]-1/embeddings[i]))

    return np.max(L_values)

time_save = np.zeros(293)
for i in range(293):
    blank_list = np.zeros(8)
    for j in range(8):
        blank_list[j] = value_save[j][i]/embeddings[j]
    time_save[i]=estimate_Lipschitz_constant(blank_list)

plt.plot(x_new, time_save)
plt.xlim(0,300)
plt.show()
