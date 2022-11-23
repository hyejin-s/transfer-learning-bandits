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

repetition = 100
over_num = list()
A_sorted = np.sort(estimated_list)[::-1]
for i in range(repetition):
    if estimated_list[i] in A_sorted[:len(under_num)]:
        over_num.append(i)

assert len(over_num) == len(under_num), "check number of over_num"  

horizon = 50000
### Estimated over time ###
with open(os.path.join(dir, "estimated_Lipschitz_Constant.txt")) as f:
    data = f.read().split()
estimated_LC_top = np.zeros(horizon)
estimated_LC_bottom = np.zeros(horizon)
for i in range(repetition):
    if int(data[((horizon+1)*2)*i+1]) in over_num:
        for j in range(horizon):
            estimated_LC_top[j] += float(data[((horizon+1)*2)*i+1+(2*j+2)])
    if int(data[((horizon+1)*2)*i+1]) in under_num:
        for j in range(horizon):
            estimated_LC_bottom[j] += float(data[((horizon+1)*2)*i+1+(2*j+2)])
        
for j in range(horizon):
    estimated_LC_top[j] = estimated_LC_top[j] / len(under_num)
    estimated_LC_bottom[j] = estimated_LC_bottom[j] / len(under_num)

plt.figure(figsize=(20,15))
X = np.linspace(0,horizon,horizon, endpoint=False)

plt.plot(X, estimated_LC_top, color='b', label="top 5.0%", lw=5)
plt.plot(X, estimated_LC_bottom, color='r', label="bottom 5.0%", lw=5)
plt.plot(X, [200 for i in range(horizon)], color='g', label="true Lipschitz constant", lw=4)

plt.xlabel('Time', fontsize=60)
plt.ylabel('Lipschitz Constant', fontsize=60)
plt.ylim(-10,250)
plt.xticks(fontsize=40)
plt.xticks([0,10000,20000,30000,40000,50000], ['0', '10k', '20k', '30k', '40k', '50k'])
plt.yticks(fontsize=40)
plt.yticks([0.1,50,100,150,200,250], ['0.1', '50', '100', '150', '200', '250'])
plt.legend(fontsize=40)

plt.savefig(dir+"/estimated_over_time", dpi=300)
plt.savefig(dir+"/estimated_over_time.pdf", format='pdf', dpi=300) 