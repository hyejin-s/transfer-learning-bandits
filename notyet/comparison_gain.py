name = ['/Dataset 0.csv','/Dataset 1.csv','/Dataset 2.csv','/Dataset 3.csv',
        '/Dataset 4.csv','/Dataset 5.csv','/Dataset 6.csv','/Dataset 7.csv']

import numpy as np 
import matplotlib.pyplot as plt 
from scipy import interpolate 
from scipy.optimize import linprog


from math import log


embeddings = [54, 48, 36, 24, 18, 12, 9, 6]

sliding_window = 5
num = 293
M = num - sliding_window + 1 # the ordering of Transfer Learning

# arm info
value_save = np.zeros((8,num))
for i in range(8):
    data = np.loadtxt(r'/home/phj/bandits/data'+name[i], delimiter=',', dtype = np.float32)     
    x_ori = data[:,0] 
    y_ori = data[:,1] 
    f1 = interpolate.interp1d(x_ori,y_ori) 
    x_new = np.linspace(3,296,num=293,endpoint=True)
    value_save[i]=f1(x_new)
    undervalue = np.where(value_save[i] < 0)[0]
    for j in range(len(undervalue)):
        value_save[i][undervalue[j]] = 0

for i in range(8):
    value_save[i] = value_save[i]/embeddings[i]   # success rate

arm_info = np.zeros((M, len(embeddings))) # thetas
for i in range(len(embeddings)):
    for m in range(M):
        arm_info[m][i] = np.sum(value_save[i][m:m+sliding_window])/sliding_window

mu_info = np.zeros((M, len(embeddings))) # mu
for m in range(M):
    for i in range(len(embeddings)):
        mu_info[m][i] = arm_info[m][i] * embeddings[i]


def com_Lipschitz_constant(thetas, embeddings):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs((thetas[i+1]-thetas[i])/(embeddings[i+1]-embeddings[i])))

    return np.amax(L_values)


Ltrue_info = np.zeros(M)
Lmutrue_info = np.zeros(M)
for m in range(M):
    Ltrue_info[m] = com_Lipschitz_constant(arm_info[m], embeddings)
    Lmutrue_info[m] = com_Lipschitz_constant(mu_info[m], embeddings)


# print(arm_info)
# print(mu_info)
# print(Ltrue_info)

eps = 1e-15  #: Threshold value: everything in [0, 1] is truncated to [eps, 1 - eps]

# Bernoulli KL-divergence
def klBern(x, y):
   x = min(max(x, eps), 1 - eps)
   y = min(max(y, eps), 1 - eps)
   return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))
klBern_vect = np.vectorize(klBern)

def log_plus(x):
    return max(1, log(x))

def solve_optimization_problem__classic(mu, thetas):
    values = 0
    mu_max = np.max(mu)
    theta_max = np.max(thetas)

    for i, mu_value in enumerate(mu):
        if mu_value < mu_max: # sub-optimal arms
            # values += (theta_max - thetas[i]) / klBern(thetas[i], mu_max/embeddings[i])
            values += (mu_max - mu[i]) / (klBern(thetas[i], mu_max/embeddings[i]))
            # values += (mu_max - mu[i]) / (embeddings[i]*klBern(thetas[i], mu_max/embeddings[i]))
    return values

def get_confusing_bandit(j, L, mu, thetas, n): 
    # values : \lambda_i^k (arm k,i \in K^{-})
    lambda_values = np.zeros_like(thetas)
    mu_max = np.max(mu)
    for i, theta in enumerate(thetas):
        if i == j:
            lambda_values[i] = mu_max/embeddings[i]
        else:
            if n == 0: # theta
                lambda_values[i] = max(theta, mu_max/embeddings[i]-L*abs(embeddings[k]-embeddings[i]))
            else: # mu
                lambda_values[i] = max(mu[i], mu_max-L*abs(embeddings[j]-embeddings[i]))/embeddings[i]
            # if lambda_values[i]>1:
            #     lambda_values[i] = theta/r[i]
    return lambda_values

def get_confusing_bandit_d(j, d, mu, thetas, n):
    # values : \lambda_i^k (arm k,i \in K^{-})
    lambda_values = np.zeros_like(thetas)
    mu_max = np.max(mu)
    for i, theta in enumerate(thetas):
        if i == j:
            lambda_values[i] = mu_max/embeddings[i]
        else:
            if n == 0: # theta
                lambda_values[i] = max(theta, mu_max/embeddings[i] - d[i][j])
            else: # mu
                lambda_values[i] = max(mu[i], mu_max-d[i][j])/embeddings[i]
            # if lambda_values[i]>1:
            #     lambda_values[i] = theta/r[i]
    return lambda_values   

def solve_optimization_problem__Lipschitz(mu, thetas, L, n): # theta: n == 0, mu: n == 1
    mu_max = np.max(mu)
    c = mu_max - mu  # c : (\theta^*-theta_k)_{k\in K}
    
    sub_arms = (np.nonzero(c))[0]
    opt_arms = (np.where(c==0))[0]
    
    A_ub=np.zeros((sub_arms.size, sub_arms.size))
    for j, k in enumerate(sub_arms):
        nu = get_confusing_bandit(k, L, mu, thetas, n)# get /lambda^k
        for i, idx in enumerate(sub_arms):         # A_eq[j]=
            A_ub[j][i] = klBern(thetas[idx], nu[idx])
            # if n == 0:
                # A_ub[j][i] = klBern(thetas[idx], nu[idx])
            # else:
                # A_ub[j][i] = embeddings[idx] * klBern(thetas[idx], nu[idx])    
    A_ub = (-1)*A_ub
    b_ub = (-1)*np.ones_like(np.arange(sub_arms.size, dtype=int))
    delta = c[c!=0]

    bounds_sub = np.zeros((sub_arms.size, 2))
    for idx, i in enumerate(sub_arms):
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

def solve_optimization_problem__Lipschitz_d(mu, thetas, d, n):
    mu_max = np.max(mu)
    c = mu_max - mu  # c : (\theta^*-theta_k)_{k\in K}
    
    sub_arms = (np.nonzero(c))[0]
    opt_arms = (np.where(c==0))[0]
    
    A_ub=np.zeros((sub_arms.size, sub_arms.size))
    for j, k in enumerate(sub_arms):
        nu = get_confusing_bandit_d(k, d, mu, thetas, n) # get /lambda^k
        for i, idx in enumerate(sub_arms):         # A_eq[j]=
            # A_ub[j][i] = klBern(thetas[idx], nu[idx])
            if n == 0:
                A_ub[j][i] = klBern(thetas[idx], nu[idx])
            else:
                A_ub[j][i] = embeddings[idx] * klBern(thetas[idx], nu[idx])    

    A_ub = (-1)*A_ub
    b_ub = (-1)*np.ones_like(np.arange(sub_arms.size, dtype=int))
    delta = c[c!=0]

    bounds_sub = np.zeros((sub_arms.size, 2))
    for idx, i in enumerate(sub_arms):
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
        v = 0
        for i in range(8):
            if i in sub_arms:
                v += result[i] * (mu_max - mu[i])
            else:
                v += mu_max / embeddings[i] * mu[i]
        print("v!!!!!!!!!!!!!")
        print(v)
        print("res.fun!!!!!!!!!!!!!!!!!!!")
        print(res.fun)
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

plotting_value = np.zeros((5, M))

d_matrix_mu = np.zeros((8,8))
d_matrix_theta = np.zeros((8,8))

for m in range(M):
    print(m)
    for j in range(8):
        for k in range(8):
            if d_matrix_mu[j][k] < abs(embeddings[j]*arm_info[m][j]-embeddings[k]*arm_info[m][k]):
                d_matrix_mu[j][k] = abs(embeddings[j]*arm_info[m][j]-embeddings[k]*arm_info[m][k])
            if d_matrix_theta[j][k] < abs(arm_info[m][j]-arm_info[m][k]):
                d_matrix_theta[j][k] = abs(arm_info[m][j]-arm_info[m][k])
    
    plotting_value[0][m] = solve_optimization_problem__classic(mu_info[m], arm_info[m])
    plotting_value[1][m] = solve_optimization_problem__Lipschitz(mu_info[m], arm_info[m], np.max(Ltrue_info), 0)
    plotting_value[2][m] = solve_optimization_problem__Lipschitz(mu_info[m], arm_info[m], np.max(Lmutrue_info), 1)
    plotting_value[3][m] = solve_optimization_problem__Lipschitz_d(mu_info[m], arm_info[m], d_matrix_theta, 0)
    plotting_value[4][m] = solve_optimization_problem__Lipschitz_d(mu_info[m], arm_info[m], d_matrix_mu, 1)

    # classic = solve_optimization_problem__classic(mu_info[m], arm_info[m])
    # lip_theta_L = solve_optimization_problem__Lipschitz(mu_info[m], arm_info[m], L=Ltrue_info[m], 0)
    # lip_mu_L = solve_optimization_problem__Lipschitz(mu_info[m], arm_info[m], L=Ltrue_info[m], 1)
    # lip_theta_d = solve_optimization_problem__Lipschitz_d(mu_info[m], arm_info[m], d_matrix, 0)
    # lip_mu_d = solve_optimization_problem__Lipschitz_d(mu_info[m], arm_info[m], d_matrix, 1)


filepath = r'/home/phj/bandits/'
labels = ['classic', 'lip_theta_L', 'lip_mu_L', 'lip_theta_d', 'lip_mu_d']

X = list(range(1, M+1))

plt.figure()
plt.plot(X, plotting_value[0], label="{}".format(labels[0]), linewidth = 1)
plt.plot(X, plotting_value[1], label="{}".format(labels[1]), linewidth = 1)
plt.plot(X, plotting_value[2], label="{}".format(labels[2]), linewidth = 1)
plt.plot(X, plotting_value[3], label="{}".format(labels[3]), linewidth = 1)
plt.plot(X, plotting_value[4], label="{}".format(labels[4]), linewidth = 1)
plt.legend()
plt.savefig(filepath+'/figure-all', dpi=300)
plt.savefig(filepath+'/figure-all.pdf', format='pdf', dpi=300)
print("Success to save figure-all")

# cum
plt.figure()
plt.plot(X, np.cumsum(plotting_value[0]), label="{}".format(labels[0]), linewidth = 1)
plt.plot(X, np.cumsum(plotting_value[1]), label="{}".format(labels[1]), linewidth = 1)
plt.plot(X, np.cumsum(plotting_value[2]), label="{}".format(labels[2]), linewidth = 1)
plt.plot(X, np.cumsum(plotting_value[3]), label="{}".format(labels[3]), linewidth = 1)
plt.plot(X, np.cumsum(plotting_value[4]), label="{}".format(labels[4]), linewidth = 1)
plt.legend()
plt.savefig(filepath+'/figure-all-cum', dpi=300)
plt.savefig(filepath+'/figure-all-cum.pdf', format='pdf', dpi=300)
print("Success to save figure-all-cum")




# # 0, 2, 4 mu
# plt.figure()
# plt.plot(X, plotting_value[0], label="{}".format(labels[0]), linewidth = 1)
# plt.plot(X, plotting_value[2], label="{}".format(labels[2]), linewidth = 1)
# plt.plot(X, plotting_value[4], label="{}".format(labels[4]), linewidth = 1)
# plt.legend()
# plt.savefig(filepath+'/figure-mu', dpi=300)
# plt.savefig(filepath+'/figure-mu.pdf', format='pdf', dpi=300)
# print("Success to save figure-mu")

# # 1, 3 theta
# plt.figure()
# plt.plot(X, plotting_value[1], label="{}".format(labels[1]), linewidth = 1)
# plt.plot(X, plotting_value[3], label="{}".format(labels[3]), linewidth = 1)
# plt.legend()
# plt.savefig(filepath+'/figure-theta', dpi=300)
# plt.savefig(filepath+'/figure-theta.pdf', format='pdf', dpi=300)
# print("Success to save figure-theta")

# # 0, 2, 4 mu - cum
# plt.figure()
# plt.plot(X, np.cumsum(plotting_value[0]), label="{}".format(labels[0]), linewidth = 1)
# plt.plot(X, np.cumsum(plotting_value[2]), label="{}".format(labels[2]), linewidth = 1)
# plt.plot(X, np.cumsum(plotting_value[4]), label="{}".format(labels[4]), linewidth = 1)
# plt.legend()
# plt.savefig(filepath+'/figure-mu-cum', dpi=300)
# plt.savefig(filepath+'/figure-mu-cum.pdf', format='pdf', dpi=300)
# print("Success to save figure-mu-cum")

# # 1, 3 theta - cum
# plt.figure()
# plt.plot(X, np.cumsum(plotting_value[1]), label="{}".format(labels[1]), linewidth = 1)
# plt.plot(X, np.cumsum(plotting_value[3]), label="{}".format(labels[3]), linewidth = 1)
# plt.legend()
# plt.savefig(filepath+'/figure-theta-cum', dpi=300)
# plt.savefig(filepath+'/figure-theta-cum.pdf', format='pdf', dpi=300)
# print("Success to save figure-theta-cum")

# plt.figure()
# X = list(range(1, M+1))
# for i in range(5):
#     # plt.figure()
#     plt.plot(X, plotting_value[i], label="{}".format(labels[i]), linewidth = 1)
#     # plt.legend()
#     # plt.show()
# plt.legend()
# plt.show()