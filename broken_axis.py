import matplotlib.pyplot as plt
import numpy as np

from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='rbgcmyk')

filepath = r"C:\Users\parke\last_data"

from collections import defaultdict
with open(r"C:\Users\parke\last_data\estimated_Lipschitz_Constant_last.txt") as f:
    data = f.read().split()
# print(data)
Y = []
Y_sort = []
for i in range(100):
    Y.append([i, float(data[i*3+2])])
    Y_sort.append(float(data[i*3+2]))
Y_sort.sort()

min_list = []
max_list = []

for i in range(len(Y_sort)):
    if Y[i][1] in Y_sort[0:10]:
        min_list.append(Y[i][0])
    if Y[i][1] in Y_sort[-11:-1]:
        max_list.append(Y[i][0])

# print(Q)
print(min_list)
print(max_list)

## read num arm pulls

with open(r"C:\Users\parke\last_data\num_ArmPulls_last.txt") as f:
    data = f.read().split()
bottom_list = []
top_list = []
for j in range(6):
    a, b = 0, 0
    for i in range(100):
        if int(data[8*i+1][:-1]) in min_list:
            # print(data[8*i+(1+j)][:-1])
            a += float(data[8*i+1+(j+1)][:-1])
            # print(a)
        if int(data[8*i+1][:-1]) in max_list:
            # print(data[8*i+(1+j)][:-1])
            b += float(data[8*i+1+(j+1)][:-1])
            # print(b)
    bottom_list.append(a/10)
    top_list.append(b/10) 

print(bottom_list)
print(top_list)

bottom=[49895.9, 42.4, 6.4, 6.7, 6.3, 42.3]
top=[735.0, 171.9, 173.4, 48575.1, 171.9, 172.7]
# bottom=[4.831575e+04, 4.095000e+01, 1.149900e+03, 7.100000e+00, 4.314000e+02, 5.490000e+01]
# top=[2407.9, 52.95, 66.15, 46656.3, 764.2, 52.5]
X=[1,2,3,4,5,6] 
X1 = []
X2 = []

width = 0.4 # the width of the bars
for i in range(len(X)):
    X1.append(X[i]+0.2)
    X2.append(X[i]-0.2)

f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
# plot the same data on both axes
ax.bar(X2, bottom, width, label="bottom 10.0%")
ax2.bar(X2, bottom, width)

ax.set_ylim(40000, 50000) # outliers only
ax2.set_ylim(0, 5000) # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

# plot the same data on both axes

ax.bar(X1, top, width, label="top 10.0%")
ax2.bar(X1, top, width)

ax.set_ylim(40000, 50000) # outliers only
ax2.set_ylim(0, 5000) # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1-d, 1+d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)  # bottom-left diagonal
ax2.plot((1-d, 1+d), (1-d, 1+d), **kwargs)  # bottom-right diagonal

ax.legend(fontsize=12)


ax.set_yticklabels(['40k','','','','','50k'], fontsize=12)
ax2.set_yticklabels([0,'','','','','5k'], fontsize=12)
ax2.set_xticklabels(['','1','2','3','4','5','6'], fontsize=11)
plt.xlabel('Arm Index', fontsize=20)
plt.ylabel('                  Number of Pulls', fontsize=20)

plt.savefig(filepath+'/topbottom_broken', dpi=300)
plt.savefig(filepath+'/topbottom__broken.pdf', format='pdf', dpi=300) 



# Q = defaultdict(list)
# Q_sort = defaultdict(list)
# for i in range(100):
#     a = Y[i] // (float((max(Y)+1)/10))
#     Q[a].append([i, float(Y[i])])
#     Q_sort[a].append(float(Y[i]))

# Q_list = list(Q_sort.values())
# Q_list[0].sort()
# Q_list[1].sort(reverse=True)

# for i in range(len(Q_sort[0])):
#     if Q[0][i][1] in Q_list[0][0:10]:
#         min_list.append(Q[0][i][0])
# for i in range(len(Q_sort[9])):
#     if Q[9][i][1] in Q_list[1][0:10]:
#         max_list.append(Q[9][i][0])
