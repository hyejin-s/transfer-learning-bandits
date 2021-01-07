name = ['\Dataset 0.csv','\Dataset 1.csv','\Dataset 2.csv','\Dataset 3.csv', '\Dataset 4.csv','\Dataset 5.csv','\Dataset 6.csv','\Dataset 7.csv']

import numpy as np 
import matplotlib.pyplot as plt 
from scipy import interpolate 
from scipy.optimize import linprog

value_save = np.zeros((8,293))
for i in range(8):
    data = np.loadtxt(r'C:\Users\parke\OneDrive\바탕 화면\Simulation_Note\OptimalRateSampling_graph'+name[i], delimiter=',', dtype = np.float32) 
    x_ori = data[:,0] 
    y_ori = data[:,1] 
    f1 = interpolate.interp1d(x_ori,y_ori) 
    x_new = np.linspace(3,296,num=293,endpoint=True)
    value_save[i]=f1(x_new)

from math import log
rates = [54, 48, 36, 24, 18, 12, 9, 6]
embeddings = [54, 48, 36, 24, 18, 12, 9, 6]

# embeddings = [1/54, 1/48, 1/36, 1/24, 1/18, 1/12, 1/9, 1/6]

def estimate_Lipschitz_constant(thetas):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i]-thetas[i+1])/abs(1/embeddings[i+1]-1/embeddings[i]))

    return np.max(L_values)

time_save = np.zeros(len(x_new))
for i in range(len(x_new)):
    blank_list = np.zeros(8)
    for j in range(8):
        blank_list[j] = value_save[j][i]/embeddings[j]
    time_save[i]=estimate_Lipschitz_constant(blank_list)

# plt.plot(x_new, time_save)
# plt.xlim(0,300)
# plt.show()

eps = 1e-15 #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]
# Binominal KL-divergence
def klBino(n, x, y):
   x = min(max(x, eps), 1 - eps)
   y = min(max(y, eps), 1 - eps)
   return n * (x*log(x/y) + (1-x)*log((1-x)/(1-y)))

def solve_optimization_problem__classic(n, r, thetas): # thetas: throughput
    values = 0
    theta_max = np.max(thetas)

    for i, theta in enumerate(thetas):
        if theta < theta_max:
            values += (theta_max - theta) / (r[i]*klBino(n, theta/r[i], theta_max/r[i]))

    return values

def get_confusing_bandit(j, d, r, thetas):
    theta_max = np.amax(thetas)
    # values : \lambda_i^k (arm k,i \in K^{-})
    lambda_values = np.zeros_like(thetas)
    for i, theta in enumerate(thetas):
        if i == j:
            lambda_values[i] = theta_max/r[i]
        else:
            lambda_values[i] = max(min(theta, theta_max/r[i] + d[i][j]), theta_max/r[i] - d[i][j])
            if lambda_values[i]>1:
                lambda_values[i] = theta/r[i]

    return lambda_values

def solve_optimization_problem__Lipschitz(n, d, r, thetas, L):
    theta_max = np.max(thetas)
    c = theta_max - thetas  # c : (\theta^*-theta_k)_{k\in K}
    
    sub_arms = (np.nonzero(c))[0]
    opt_arms = (np.where(c==0))[0]

    A_ub=np.zeros((sub_arms.size, sub_arms.size))
    for j, k in enumerate(sub_arms):
        #nu = get_confusing_bandit(k, rates, L, thetas) # get /lambda^k
        nu = get_confusing_bandit(k, d, rates, thetas)
        for i, idx in enumerate(sub_arms):         # A_eq[j]=
            A_ub[j][i] = r[idx] * klBino(n, thetas[idx]/r[idx], nu[idx])
    A_ub = (-1)*A_ub
    b_ub = (-1)*np.ones_like(np.arange(sub_arms.size, dtype=int))
    delta = c[c!=0]

    bounds_sub = np.zeros((sub_arms.size, 2))
    for idx, i in enumerate(np.where(thetas != max(thetas))[0]):
        bounds_sub[idx] = (0, None)

    ## revised simplex
    try:
        res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='revised simplex', bounds=bounds_sub)
    except Exception as e:
        print(str(e))
        res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=bounds_sub)
        if res.success == True: 
            print("LinearProgramming Error_Exception: success")
        else:
            print("LinearProgramming Error_Exception: fail") 
            return np.full(thetas.size, -1)

    if res.success == True: # return res.x
        result = np.zeros(thetas.size)
        for i, idx in enumerate(opt_arms):
            result[idx] = np.inf
        for i, idx in enumerate(sub_arms):
            result[idx] = res.x[i]
        #return result
        return res.fun
    else: # Fail
        if res.status == 2: # we can ignore this failure
            result = np.zeros(thetas.size)
            for i, idx in enumerate(opt_arms):
                result[idx] = np.inf
            for i, idx in enumerate(sub_arms):
                result[idx] = res.x[i]
            return res.fun
        elif res.status == 4: # numerical difficult error
            # option_again = {'tol':1e-8, 'sym_pos':False, 'cholesky':False, 'lstsq':True}
            print("LinearProgramming Error: Numerical difficulties error")
            res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='interior-point', bounds=bounds_sub)
            # res = linprog(delta, A_ub=A_ub, b_ub=b_ub, method='revised simplex', options=option_again)
            if res.success == True: 
                print("LinearProgramming Error4: success")
            else: 
                print("LinearProgramming Error4: fail")
                return np.full(thetas.size, -1)
            
            result = np.zeros(thetas.size)
            for i, idx in enumerate(opt_arms):
                result[idx] = np.inf
            for i, idx in enumerate(sub_arms):
                result[idx] = res.x[i]
            return res.fun
        else:
            print("LinearProgramming Error: Last fail")
            return np.full(thetas.size, -1)


thetas = np.transpose(value_save) # 293 by 8
success = np.zeros((293,8))

#######################################
############# for 1-dimension embedding

from scipy.optimize import minimize
from math import sqrt

d_matrix_max = np.zeros((8,8))
initial_1d = np.zeros((8))
d_matrix = np.zeros((8,8))
embeddings_1d = np.zeros((293, 8))

for i in range(len(x_new)):
    for j in range(8): # to change success probability
        success[i][j] = thetas[i][j] / rates[j]
    for j in range(8):
        for k in range(8):
            if d_matrix_max[j][k] < abs(success[i][j]-success[i][k]):
                d_matrix_max[j][k] = abs(success[i][j]-success[i][k])
    if i == 0:
        initial_1d = d_matrix_max[0]

    def objective(x):
        x[0] = 0
        for i in range(len(x)):
            for j in range(len(x)):
                d_matrix[i][j] = abs(d_matrix_max[i][j]-(x[i]-x[j]))
        return np.sum(d_matrix)

    sol = minimize(objective, initial_1d, method='SLSQP')
    # print(d_matrix_max)
    # print(sol.x)
    embeddings_1d[i] = sol.x

filepath = r"C:\Users\parke\OneDrive\바탕 화면\Simulation_Note\fig"

plt.figure()
embeddings_1d_t = np.transpose(embeddings_1d)
for i in range(8):   
    plt.plot(embeddings_1d_t[i], np.linspace(1,293,293), label='{}'.format(i))
plt.legend()
plt.savefig(filepath+"/figure_1d_path", dpi=300)

plt.figure()
embeddings_1d_t = np.transpose(embeddings_1d)
for i in range(8):   
    plt.scatter(embeddings_1d_t[i][292], 0, label='{}'.format(i))
plt.legend()
plt.savefig(filepath+"/figure_1d_last", dpi=300)

d_matrix = np.zeros((8,8))
d_matrix_max = np.zeros((8,8))
initial_2d = np.zeros((2,8))
embeddings_2d = np.zeros((293,8,2))

for i in range(len(x_new)):
    print(i)
    for j in range(8): # to change success probability
        success[i][j] = thetas[i][j] / rates[j]
    for j in range(8):
        for k in range(8):
            if d_matrix_max[j][k] < abs(success[i][j]-success[i][k]):
                d_matrix_max[j][k] = abs(success[i][j]-success[i][k])
    if i == 0:
        initial_2d[0] = d_matrix_max[0]
        initial_2d[1] = d_matrix_max[0]

    def objective_2d(x):
        x = np.reshape(x, (int(len(x)/2), 2))
        for i in range(len(x)):
            for j in range(len(x)):
                d_matrix[i][j] = abs(d_matrix_max[i][j]-sqrt((x[i][0]-x[j][0])**2 + (x[i][1]-x[j][1])**2))
        return np.sum(d_matrix)

    sol_2d = minimize(objective_2d, np.transpose(initial_2d), method='SLSQP', jac='2-point')
    sol_reshape = np.reshape(sol_2d.x, (int(len(sol_2d.x)/2),2))
    # print(d_matrix_max)
    # print(sol.x)
    embeddings_2d[i] = sol_reshape

print(d_matrix_max)
plt.figure()
embeddings_2d_t = np.transpose(embeddings_2d, (1,2,0)) #(8,2,293)
for i in range(8):   
    plt.plot(embeddings_2d_t[i][0], embeddings_2d_t[i][1], label='{}'.format(i), )
    # plt.scatter(embeddings_2d_t[i][0][292], embeddings_2d_t[i][1][292], label='{}'.format(i), )
plt.legend()
plt.savefig(filepath+"/figure_2d_path", dpi=300)

plt.figure()
embeddings_2d_t = np.transpose(embeddings_2d, (1,2,0)) #(8,2,293)
for i in range(8):   
    plt.scatter(embeddings_2d_t[i][0][292], embeddings_2d_t[i][1][292], label='{}'.format(i), )
plt.legend()
plt.savefig(filepath+"/figure_2d_last", dpi=300)


