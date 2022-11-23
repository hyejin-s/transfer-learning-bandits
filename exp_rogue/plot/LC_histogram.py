import os
import numpy as np
import matplotlib.pyplot as plt

dir = "./3" 

### estimated Lipschitz Constant list
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

### Histogram of \hat L 
plt.figure(figsize=(20,15))
plt.hist(estimated_list, bins=10, color='r', rwidth=0.9)

### write font in plot
x = [10, 194.5]
y1 = [6, 93.7]
y = [np.round(np.mean(under_lc),4), np.round(np.mean(over_lc),4)]
for i, v in enumerate(x):
    plt.text(v, y1[i], y[i],  
             fontsize = 40, 
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')

### plotting
plt.xlabel('Estimated Lipschitz Constant', fontsize=60)
plt.ylabel('Number of Repetitions', fontsize=60)
plt.xticks(fontsize=40)
plt.yticks(fontsize=40)

plt.savefig(dir+"/histogram_LC", dpi=300)
plt.savefig(dir+"/histogram_LC.pdf", format='pdf', dpi=300) 

