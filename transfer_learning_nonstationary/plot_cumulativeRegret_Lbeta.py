import numpy as np
import matplotlib.pyplot as plt
import re
sliding_window = 5
episode_length = 289
M = episode_length - sliding_window 
smooth = 1


valuelist = [[0.1, 0.1], [0.3, 0.1], [0.5, 0.1], [0.1, 0.05], [0.3, 0.05], [0.5, 0.05], [0.1, 0.01], [0.3, 0.01], [0.5, 0.01]] # [beta, epsilon]
nbPolicies = 9
num_list = []

# colors = ['tomato', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple', 'navajowhite', 'lavender', 'darkgrey', 'royalblue', 'darkorange', 'orange'] # 13
# labels = [r'$\pi(\infty)$', r'$d:(\delta, \varepsilon_\delta)=(0.1, 0.05)$', r'$d:(\beta, \varepsilon_\delta)=(0.3, 0.05)$', r'$L:(\beta, \varepsilon_\beta)=(0.1, 0.05)$', r'$L:(\beta, \varepsilon_\beta)=(0.3, 0.05)$', r'$\pi({d}_{true})$', r'$\pi({L}_{true})$']
labels = [r'$\pi(\infty)$', r'$L:(\beta, \varepsilon_\beta)=(0.1, 0.05)$', r'$L:(\beta, \varepsilon_\beta)=(0.3, 0.05)$'
, r'$d:(\delta, \varepsilon_\delta)=(0.1, 0.05)$', r'$d:(\delta, \varepsilon_\delta)=(0.3, 0.05)$', r'$\pi({L}_{true})$', r'$\pi({d}_{true})$']

colors = ['tab:cyan', 'crimson', 'palevioletred', 'tab:blue',  'limegreen', 'tab:orange', 'rebeccapurple', 'tab:blue', 'limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'lavender', 'rebeccapurple', 'navajowhite', 'lavender', 'darkgrey', 'royalblue', 'orange'] # 11
# labels = ['inf', '(0.1, 0.1)', '(0.3, 0.1)', '(0.5, 0.1)', '(0.1, 0.05)', '(0.3, 0.05)', '(0.5, 0.05)', '(0.1, 0.5)', '(0.3, 0.5)', '(0.5, 0.5)', 'true']
# labels = [r'$\pi(\infty)$', r'$(\beta, \epsilon)=(0.1, 0.1)$', r'$(\beta, \epsilon)=(0.3, 0.1)$', r'$(\beta, \epsilon)=(0.5, 0.1)$', r'$(\beta, \epsilon)=(0.1, 0.05)$'
            # , r'$(\beta, \epsilon)=(0.3, 0.05)$', r'$(\beta, \epsilon)=(0.5, 0.05)$', r'$(\beta, \epsilon)=(0.1, 0.01)$', r'$(\beta, \epsilon)=(0.3, 0.01)$', r'$(\beta, \epsilon)=(0.5, 0.01)$', r'$\pi(\hat{L}_m)$']
# filepath = r"C:\Users\parke\bandit\checking"
# colors = ['tab:cyan', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:black']
with open(r"/home/phj/bandits/non-stationary-d-Lipschitz-true-d/regret_transfer.txt") as f:
    data = f.read().split()
    plt.figure(figsize=(24,16))
    X = list(range(1, int(M)+1))
    for i in range(nbPolicies):
        regret_list = []
        for m in range(M):
            if data[(nbPolicies+1)*m+(i+1)][-1] == ',':
                regret_list.append(float(data[(nbPolicies+1)*m+(i+1)][:-1]))            
            else:
                regret_list.append(float(data[(nbPolicies+1)*m+(i+1)]))
        if i in [0, 4, 5]:
            num_list.append(i)
            plt.plot(X, np.cumsum(regret_list), label=r"{}".format(labels[len(num_list)-1]), color=colors[len(num_list)-1], linewidth=7)
    for i in range(nbPolicies):
        regret_list = []
        for m in range(M):
                if data[(nbPolicies+1)*m+(i+1)][-1] == ',':
                    regret_list.append(float(data[(nbPolicies+1)*m+(i+1)][:-1]))            
                else:
                    regret_list.append(float(data[(nbPolicies+1)*m+(i+1)]))
        if i in [1, 2, 7, 8]:
            num_list.append(i)
            plt.plot(X, np.cumsum(regret_list), label=r"{}".format(labels[len(num_list)-1]), color=colors[len(num_list)-1], linewidth=7)
            
            

            
# with open(r"C:\Users\parke\bandit\checking\regret_transfer5.txt") as f:
#     data = f.read().split()
#     X = list(range(1, int(M)+1))
#     for i in range(6):
    
#         regret_list5 = []
#         for m in range(M):
#             if data[7*m+(i+1)][-1] == ',':
#                 regret_list5.append(float(data[7*m+(i+1)][:-1]))            
#             else:
#                 regret_list5.append(float(data[7*m+(i+1)]))
#         plt.plot(X, np.cumsum(regret_list5), label=r"{}, $L=5$".format(labels[i]), color=colors[i], linestyle='dashed', linewidth=5)


plt.ylabel('Cumulative Regret', fontsize=60)
plt.xlabel('Episode', fontsize=60)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)
plt.legend(fontsize=40)

plt.savefig('./d-L-cumregret', dpi=300)
plt.savefig('./d-L-cumregret.pdf', format='pdf', dpi=300) 

