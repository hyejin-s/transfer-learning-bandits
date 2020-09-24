import matplotlib.pyplot as plt
import numpy as np
import random


class GenerativeModel(object):
    def __init__(self, num_arm):
        self.num_arm = num_arm

    def gene_embedding(self):
        embeddings = list(np.linspace(0,1,self.num_arm))
        return embeddings

    def gene_arms(self, LipC):
        arms = np.zeros(self.num_arm)
        arms[0] = random.random()
        for i in range(self.num_arm-1):
        # max(a-L/(K-1), 0), min(a+L(K-1),1)
            arms[i+1] = random.uniform(max(arms[i]-LipC/(self.num_arm-1),0),min(arms[i]+LipC/(self.num_arm-1),1))
        return arms

"""
a = GenerativeModel(5)
print(a.gene_arms(10))
print(a.gene_embedding())
"""

"""
plt.figure()
X = embeddings_gene
Y = arms_gene
plt.plot(embeddings_gene, arms_gene)
plt.show()
"""