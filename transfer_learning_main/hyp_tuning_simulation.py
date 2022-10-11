import os
import numpy as np 
import argparse
from tqdm import tqdm
from math import ceil
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from scipy.optimize import curve_fit

from policies.Bernolli import Bernoulli
from transfer_learning_main.environments.evaluator_transfer_learning import *
from OSSB_transfer_hyp10 import LipschitzOSSB_DEL_true

def Lipschitz_beta(LC_list, beta, epsilon, M):
    bound_value = ceil(beta * M) # ROUND UP
    L_beta = sorted(LC_list, reverse=True)[bound_value-1] + epsilon
    return L_beta

def main(args):

    print(f"Hyperparameter tuning for Lipschitz Constant {args.L}")
    # print(f"\epsilon: {args.beta_hyp[0]}, \epsilon_\beta: {args.beta_hyp[1]}")
    
    M = args.num_episodes # the ordering of Transfer Learning
    embeddings, num_arms = args.embeddings, args.num_arms
    # beta, epsilon = args.beta_hyp[0], args.beta_hyp[1]

    ''' bandit instances '''
    with open(f'./bandit_instances/L_{args.L}/arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'rb') as f:
        episode_instances = pickle.load(f)
    
    ''' empirical Lipschitz Constance (saving in advance) '''
    with open(f'./bandit_instances/L_{args.L}/empirical_LC_arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'rb') as f:
        empirical_LC_episodes = pickle.load(f)

    # hyp ----------
    a, c, q = 0, 0.5, 1
    beta_LC = 0

    # for results ----------
    beta_episode, epsilon_episode, gamma_episode, beta_LC_episode = np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)
    regret_episode = np.zeros(M)

    for m in range(M):
        print("{} episode".format(m+1))
        
        if m == 0:
            pre_estimated_L = empirical_LC_episodes[0]
        else:
            pre_estimated_L = beta_LC
        ''' hyperparemter tuning '''
        cumulated_LC = empirical_LC_episodes[:m+1]
        max_L= np.max(cumulated_LC)
        
        q_kfold= list()
        if m < args.kfold:
            q_kfold.append(q)
        else:
            rkf = KFold(n_splits=args.kfold)
            for i in rkf.split(cumulated_LC): # kfold
                max_L_fold = np.max((cumulated_LC[i[1]]))
                q_kfold.append(max_L/max_L_fold)
            q = np.mean(q_kfold)

        # get estimated L
        estimated_L = q * pre_estimated_L

        z = np.linspace(a, estimated_L, args.linspace)
        cumul_xi_LC = [estimated_L - Lm for Lm in cumulated_LC]
        alpha_z = [len(np.where(cumul_xi_LC < i)[0]) / (m+1) for i in z]

        def func(x, gamma):
            return ((x - a)/(estimated_L - a)) ** gamma
        
        # get gamma using curve fitting and epsilon_beta using c \in (0, 1)
        gamma, _ = curve_fit(func, z, alpha_z)
        epsilon = c * pre_estimated_L

        ### if estimated_L is small, too small beta
        if gamma > 1:
            beta = ((epsilon - a)/(min(1, estimated_L) - a)) ** gamma
        else:
            beta = ((epsilon - a)/(estimated_L - a)) ** gamma
        beta = 0.01 if beta < 0.01 else 1 if beta >= 1 else beta

        # for saving
        beta_episode[m] = beta
        epsilon_episode[m] = epsilon
        gamma_episode[m] = gamma
        print(beta, epsilon, gamma)

        # transfer learning ----------
        instance = episode_instances[m]
        # print(instance)

        ENVIRONMENTS = [{"arm_type": Bernoulli, "params": instance}]
        POLICIES = []

        ''' beta '''
        beta_LC = Lipschitz_beta(cumulated_LC, beta, epsilon, m)
        beta_LC_episode[m] = beta_LC
        POLICIES.append(LipschitzOSSB_DEL_true(nbArms=num_arms,  gamma=0.001, L=beta_LC))

        configuration = {"horizon": args.horizon , "repetitions": args.repetitions, "n_jobs": args.njobs, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
        evaluation = Evaluator(configuration)
    
        for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
            evaluation.startOneEnv(envId, env)

        regret_episode[m] = evaluation.getCumulatedRegret_LessAccurate(policyId=0)[args.horizon-1]

    save_dir = os.path.join(args.save_dir, 'hyp_tuning')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.save:
        with open(f'{save_dir}/beta_episode_a_{args.a}_c_{args.c}_L_{args.L}_episodes_{M}_.pickle', 'wb') as f:
            pickle.dump(beta_episode, f)
        with open(f'{save_dir}/epsilon_episode_a_{args.a}_c_{args.c}_L_{args.L}_episodes_{M}_.pickle', 'wb') as f:
            pickle.dump(epsilon_episode, f)
        with open(f'{save_dir}/gamma_episode_a_{args.a}_c_{args.c}_L_{args.L}_episodes_{M}_.pickle', 'wb') as f:
            pickle.dump(gamma_episode, f)
        with open(f'{save_dir}/hyp_empirical_LC_a_{args.a}_c_{args.c}_arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'wb') as f:
            pickle.dump(beta_LC_episode, f)
        with open(f'{save_dir}/regret_hyp_tuning_L_a_{args.a}_c_{args.c}_{args.L}_episodes_{M}_.pickle', 'wb') as f:
            pickle.dump(regret_episode, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", help="save direction", type=str, default="./results")
    parser.add_argument("--num_arms", help="The number of arms", type=int, default=6)
    parser.add_argument("--num_episodes", help="M; the total number of transfer learning tasks", type=int, default=400)
    parser.add_argument("--L", help="Lipschitz Constant", type=float, default=0.5)
    parser.add_argument("--embeddings", help="The embedding of a bandit instance", nargs='+', type=float, default=[0, 0.8, 0.85, 0.9, 0.95, 1])
    parser.add_argument("--save", help="save results", default=True)
    
    parser.add_argument("--kfold", help="k; episode k-fold", default=5)    
    parser.add_argument("--linspace", help="The number of fraction for np.linspace", type=int, default=50)

    parser.add_argument("--horizon", help="horizon in one task", type=int, default=10000)
    parser.add_argument("--repetitions", help="repetitions for configuration", type=int, default=1)
    parser.add_argument("--njobs", help="n_jobs for configuration", type=int, default=40)
    
    parser.add_argument("--a", help="hyperparameter a", type=int, default=0)
    parser.add_argument("--c", help="hyperparameter c", type=int, default=0.5)
    
    args = parser.parse_args()

    main(args)