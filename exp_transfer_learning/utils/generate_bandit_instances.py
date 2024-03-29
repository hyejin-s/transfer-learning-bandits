import os
import numpy as np
import random
import argparse
import pickle

class GenerativeModel(object):
    def __init__(self, num_arms):
        self.num_arms = num_arms

    def gene_embedding(self, num_arms):
        #embeddings = list(np.linspace(0,1,self.num_arms)) # uniform
        embeddings = [0]
        for i in range(num_arms-1):
            embeddings.append(0.8 + i * 0.2 / (num_arms - 2))
        return embeddings

    def gene_arms(self, L):
        embeddings = []
        num = 8
        y = float(args.y)
        for i in range(num):
            embeddings.append(0)
        for i in range(num):
            embeddings.append(y)

        arms = np.zeros(len(embeddings))
        arms[0] = random.random()

        for i in range(len(embeddings)-1):
            if i == num-1 :
                arms[i+1] = random.uniform(max(arms[i]-L*abs(embeddings[i+1]-embeddings[i]),0), min(arms[i]+L*abs(embeddings[i+1]-embeddings[i]),1))
                if arms[i] >= 0.95: 
                    arms[i] = 0.95
                elif arms[i] <= 0.05:
                    arms[i] = 0.05
            else:
                arms[i+1] = arms[i]
                if arms[i] >= 0.95: 
                    arms[i] = 0.95
                elif arms[i] <= 0.05:
                    arms[i] = 0.05                
        # print(arms)
        return embeddings, arms

def create_dir(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory.'+ directory)

def main(args):
    generation = GenerativeModel(args.num_arms)
    episode_instance = list()
    for _ in range(args.num_episodes):
        embeddings, instance = generation.gene_arms(args.L)
        episode_instance.append(instance)

    if args.save:
        create_dir(args.save_dir)
        with open(f'{args.save_dir}/arms_{len(embeddings)}_y_{args.y}_L_{args.L}_{args.num}.pickle', 'wb') as f:
            pickle.dump(episode_instance, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_arms", help="The number of arms", type=int, default=6)
    parser.add_argument("--num_episodes", help="M; the total number of transfer learning tasks", type=int, default=400)
    parser.add_argument("--L", help="Lipschitz Constant", type=float, default=0.5)
    parser.add_argument("--save", help="save bandit instances in pickle file", default=False)
    parser.add_argument("--save_dir", type=str, default='./bandit_instances')
    parser.add_argument("--num", default=0)
    parser.add_argument("--y", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
