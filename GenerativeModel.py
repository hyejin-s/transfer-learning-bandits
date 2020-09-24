import matplotlib.pyplot as plt
import numpy as np
import random

# the number of arms
K = int(input('the number of arms: '))
embeddings_gene = list(np.linspace(0,1,K))


arms_gene = np.zeros(K) # first arm is random value \in [0,1]
arms_gene[0] = random.random()
LC = 10 #### not define yet
for i in range(K-1):
    # max(a-L/(K-1), 0), min(a+L(K-1),1)
    arms_gene[i+1] = random.uniform(max(arms_gene[i]-LC/(K-1),0),min(arms_gene[i]+LC/(K-1),1))

"""
plt.figure()
X = embeddings_gene
Y = arms_gene
plt.plot(embeddings_gene, arms_gene)
plt.show()
"""