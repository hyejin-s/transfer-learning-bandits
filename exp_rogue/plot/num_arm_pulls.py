import os
import numpy as np
import matplotlib.pyplot as plt

dir = "./3" 

# estimated Lipschitz Constant list (\hat{L})
estimated_list = [203.4673729833853, 203.6891758694907, 201.38304269392648, 196.18125050036008, 199.6432436765942, 201.965415098871, 199.4035705708108, 200.70450724521638, 200.86086086086067, 198.2907009327087, 201.02648409150123, 198.44287887278824, 200.34424785845792, 203.87979457550938, 201.33622246298285, 201.47759580346758, 201.51976902718727, 201.68201842210635, 0.10380692556608222, 199.28697324146754, 201.71347358728488, 199.018920812894, 202.08550156112383, 198.84639347227508, 201.17715361053817, 202.67724740491346, 202.40573212713136, 198.84639347227508, 201.24489632535406, 199.28697324146754, 200.0522854787136, 201.2173146998757, 197.7103215975618, 199.7584784140081, 0.10266255694044221, 199.15890657855195, 199.59515792849106, 204.52133517550692, 203.21822455252197, 202.70594828276342, 0.10099225749539001, 200.70450724521638, 201.65129874043254, 203.15359707165968, 203.21822455252197, 199.23179021035256, 199.4035705708108, 201.33622246298285, 198.84639347227508, 199.83583912233973, 12.658227848101255, 198.54295092466558, 201.43351953070237, 200.52881437413598, 202.02181963767373, 198.38680623661492, 198.64428534183648, 199.50364262268815, 204.49708893796407, 197.7103215975618, 200.86462252822014, 201.12559834965634, 198.0626050756543, 199.9439596509485, 202.27058847085658, 203.27172977193985, 200.86086086086067, 200.0921806741211, 202.6407004467951, 202.6228851736909, 202.67724740491346, 201.90537186774458, 200.24417580658056, 200.09630625388718, 198.52056771710352, 200.34504202692113, 202.08550156112383, 200.60443519333901, 199.51908626390124, 201.17715361053817, 198.22272035865805, 200.40429108958432, 197.7103215975618, 201.92692893197642, 0.10149387621374042, 200.06837194337174, 199.65173531884213, 198.16267712753165, 202.46577535825776, 202.3540244610363, 0.10045012132601415, 199.4035705708108, 203.15359707165968, 204.0669281882954, 199.1369794279978, 201.99835806251357, 199.59515792849106, 200.80457929709374, 201.8253142262427, 202.6228851736909]

under_num = list()
under_lc, over_lc = list(), list()
thres = 100

for i in range(len(estimated_list)):
    if estimated_list[i] < thres:
        under_num.append(i)
        under_lc.append(estimated_list[i])
    else:
        over_lc.append(estimated_list[i])

### arm pulls ###
arm_pulls_under = np.zeros(6)      
with open(os.path.join(dir, "num_ArmPulls.txt")) as f:
    data = f.read().split()
for i in range(100):
    if int(data[8*i+1]) in under_num:
        print(i)
        # arm index 더하기
        for j in range(6):
            arm_pulls_under[j] += int(data[8*i+1+(j+1)][:-1])
for j in range(6):
    arm_pulls_under[j] = arm_pulls_under[j] / len(under_num) 

repetition = 100
arm_pulls_over = np.zeros(6)
over_num = list()
A_sorted = np.sort(estimated_list)[::-1]
for i in range(repetition):
    if estimated_list[i] in A_sorted[:len(under_num)]:
        over_num.append(i)
        
for i in range(repetition):
    if int(data[8*i+1]) in over_num:
        for j in range(6):
            arm_pulls_over[j] += int(data[8*i+1+(j+1)][:-1])
            
assert len(over_num) == len(under_num), "check number of over_num"  

for j in range(6):
    arm_pulls_over[j] = arm_pulls_over[j] / len(under_num) 
print(arm_pulls_over)

### starting plot
X = [1, 2, 3, 4, 5, 6] 
X1, X2 = [], []

width = 0.4 # the width of the bars
for i in range(len(X)):
    X1.append(X[i]+0.2)
    X2.append(X[i]-0.2)

f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

# plot the same data on both axes
ax.bar(X1, arm_pulls_over, width, label="top 5.0%", color='b')
ax2.bar(X1, arm_pulls_over, width, color='b')

ax.set_ylim(49000, 50000) # outliers only
ax2.set_ylim(0, 1000) # most of the data

# hide the spines between ax and ax2
ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

ax.bar(X2, arm_pulls_under, width, label="bottom 5.0%", color='r')
ax2.bar(X2, arm_pulls_under, width, color='r')

ax.set_ylim(49000, 50000) # outliers only
ax2.set_ylim(0, 1000) # most of the data

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

ax.set_yticklabels(['49k','','','','','50k'], fontsize=12)
ax2.set_yticklabels([0,'','','','','1k'], fontsize=12)
ax2.set_xticklabels(['','1','2','3','4','5','6'], fontsize=12)
plt.xlabel('Arm Index', fontsize=20)
plt.ylabel('                         Number of Pulls', fontsize=20)

plt.savefig(dir+'/num_arm_pulls', dpi=300)
plt.savefig(dir+'/num_arm_pulls.pdf', format='pdf', dpi=300) 