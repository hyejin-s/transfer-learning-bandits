import matplotlib.pyplot as plt
import numpy as np
import random

class GenerativeModel(object):
    def __init__(self, num_arm):
        self.num_arm = num_arm

    def gene_embedding(self):
        #embeddings = list(np.linspace(0,1,self.num_arm)) # uniform
        embeddings = [0]
        for i in range(self.num_arm-1):
            embeddings.append(0.9 + i* 0.1/(self.num_arm-2))
        return embeddings

    def gene_arms(self, LipC):
        arms = np.zeros(self.num_arm)
        arms[0] = random.random()
        for i in range(self.num_arm-1):
        # max(a-L/(K-1), 0), min(a+L(K-1),1)
            arms[i+1] = random.uniform(max(arms[i]-LipC*abs(embeddings[i+1]-embeddings[i]),0),min(arms[i]+LipC*abs(embeddings[i+1]-embeddings[i]),1))
        for i in range(self.num_arm):
            if arms[i] >= 0.95: 
                arms[i] = 0.95
            if arms[i] <= 0.05:
                arms[i] = 0.05
        return arms


generation = GenerativeModel(10)
#arms = generation.gene_arms(0.2)
embeddings = generation.gene_embedding()


"""
plt.figure()
plt.ylim(0, 1)
for i in range(10):
    generation = GenerativeModel(10) #The number of arms = 10
    arms = generation.gene_arms(0.2) #Lipschitz Constant = 0.2

    plt.plot(embeddings, arms, label="{}".format(i))
plt.legend()
plt.show()
"""
