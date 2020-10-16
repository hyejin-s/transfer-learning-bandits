import matplotlib.pyplot as plt
import numpy as np

from cycler import cycler
plt.rcParams['axes.prop_cycle'] = cycler(color='rbgcmyk')

filepath = r"C:\Users\parke\OneDrive\바탕 화면\fig"

bottom=[4.831575e+04, 4.095000e+01, 1.149900e+03, 7.100000e+00, 4.314000e+02, 5.490000e+01]
top=[2407.9, 52.95, 66.15, 46656.3, 764.2, 52.5]
X=[1,2,3,4,5,6] 
X1 = []
X2 = []

width = 0.4 # the width of the bars
for i in range(len(X)):
    X1.append(X[i]+0.2)
    X2.append(X[i]-0.2)

f, (ax, ax2) = plt.subplots(2, 1, sharex=True)
# plot the same data on both axes
ax.bar(X2, bottom, width, label="bottom 20.0%")
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

ax.bar(X1, top, width, label="top 20.0%")
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

plt.savefig(filepath+'/topbottom', dpi=300)
plt.savefig(filepath+'/topbottom.pdf', format='pdf', dpi=300) 
