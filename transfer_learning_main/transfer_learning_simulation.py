import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil

from policies.Bernolli import Bernoulli
from environments.evaluator_transfer_learning import *
from OSSB_transfer_hyp10 import OSSB_DEL, LipschitzOSSB_DEL_true


def Lipschitz_beta(LC_list, beta, epsilon, M):
    bound_value = ceil(beta * M) # ROUND UP
    L_beta = sorted(LC_list, reverse=True)[bound_value-1] + epsilon
    return L_beta

def com_Lipschitz_constant(thetas, embeddings):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i+1]-thetas[i])/(embeddings[i+1]-embeddings[i]))

    return np.amax(L_values)

def main(args):
    print(f"Transfer Learning for Lipschitz Constant {args.L}, the horizon of each instance {args.horizon} ")
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    M = args.num_episodes # the ordering of Transfer Learning
    embeddings, num_arms = args.embeddings, args.num_arms
    
    ''' bandit instances '''
    with open(f'./bandit_instances/L_{args.L}/arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'rb') as f:
        episode_instances = pickle.load(f)
        
    ''' empirical Lipschitz Constance (saving in advance) '''
    with open(f'./bandit_instances/L_{args.L}/empirical_LC_arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'rb') as f:
        empirical_LC_episodes = pickle.load(f)

    eps_beta = args.L * 0.1
    hyp_list = [[0.1, eps_beta], [0.3, eps_beta], [0.5, eps_beta]]
    
    ''' hyper-parameters tuning value '''
    with open(f'{args.save_dir}/hyp_tuning/beta_episode_a_{args.a}_c_{args.c}_L_{args.L}_episodes_{M}_.pickle', 'rb') as f:
        beta_episode = pickle.load(f)
    with open(f'{args.save_dir}/hyp_tuning/epsilon_episode_a_{args.a}_c_{args.c}_L_{args.L}_episodes_{M}_.pickle', 'rb') as f:
        epsilon_episode = pickle.load(f)
    
    # for results ------
    ''' Empirical Lipschitz Constant '''
    true_LC_episode, max_LC_episode = np.zeros(M), np.zeros(M)
    beta_LC_episode = np.zeros((len(hyp_list), M))
    beta_hyp_LC_episode = np.zeros(M)
    
    ''' regret '''
    regret_episode = np.zeros((len(hyp_list)+3, M))
    regret_episode_hyp = np.zeros(M)
    
    save_path = os.path.join(args.save_dir, 'transfer_learning')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # transfer learning ----------
    for m in range(M):
        print("{} episode".format(m+1))
        
        instance = episode_instances[m]
        # generate model ----------
        ENVIRONMENTS = [{"arm_type": Bernoulli, "params": instance}]
        POLICIES = []
        
        ''' inf '''
        POLICIES.append({"archtype":OSSB_DEL, "params":{}})
        
        ''' true '''
        true_LC = com_Lipschitz_constant(instance, embeddings)
        true_LC_episode[m] = true_LC
        POLICIES.append(LipschitzOSSB_DEL_true(nbArms=6, gamma=0.001, L=true_LC))
        
        ''' max '''
        max_LC = np.max(empirical_LC_episodes[:m+1])     
        max_LC_episode[m] = max_LC
        POLICIES.append(LipschitzOSSB_DEL_true(nbArms=6, gamma=0.001, L=max_LC))
        
        ''' beta '''
        for idx in range(len(hyp_list)):
            beta_LC = Lipschitz_beta(empirical_LC_episodes[:m+1], hyp_list[idx][0], hyp_list[idx][1], m)
            beta_LC_episode[idx][m] = beta_LC
            POLICIES.append(LipschitzOSSB_DEL_true(nbArms=6, gamma=0.001, L=beta_LC))
            
        ''' hyp '''
        hyp_LC = Lipschitz_beta(empirical_LC_episodes[:m+1], beta_episode[m], epsilon_episode[m], m)
        beta_hyp_LC_episode[m] = hyp_LC
        POLICIES.append(LipschitzOSSB_DEL_true(nbArms=6, gamma=0.001, L=hyp_LC))

        # start ----------
        configuration = {"horizon": args.horizon, "repetitions": REPETITIONS, "n_jobs": args.njobs, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
        evaluation = Evaluator(configuration)
        for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
            evaluation.startOneEnv(envId, env)
        
        # save results ----------
        ''' regret '''
        for idx in range(len(POLICIES)-1):
            regret_episode[idx][m] = evaluation.getCumulatedRegret_LessAccurate(policyId=idx)[args.horizon-1]
        regret_episode_hyp[m] = evaluation.getCumulatedRegret_LessAccurate(policyId=len(POLICIES)-1)[args.horizon-1]
        
    if args.save:    
        with open(f'{save_path}/regret__arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'wb') as f:
            pickle.dump(regret_episode, f)
        with open(f'{save_path}/regret_hyp__arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'wb') as f:
            pickle.dump(regret_episode_hyp, f)
            
        with open(f'{save_path}/true_LC__arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'wb') as f:
            pickle.dump(true_LC_episode, f)
        with open(f'{save_path}/max_LC__arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'wb') as f:
            pickle.dump(max_LC_episode, f)
        with open(f'{save_path}/beta_LC__arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'wb') as f:
            pickle.dump(beta_LC_episode, f)
        with open(f'{save_path}/hyp_LC__arms_{num_arms}_episodes_{M}_L_{args.L}_.pickle', 'wb') as f:
            pickle.dump(beta_hyp_LC_episode, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", help="save direction", type=str, default="./results")
    parser.add_argument("--num_arms", help="The number of arms", type=int, default=6)
    parser.add_argument("--num_episodes", help="M; the total number of transfer learning tasks", type=int, default=400)
    parser.add_argument("--L", help="Lipschitz Constant", type=float, default=0.5)
    parser.add_argument("--embeddings", help="The embedding of a bandit instance", nargs='+', type=float, default=[0, 0.8, 0.85, 0.9, 0.95, 1])
    parser.add_argument("--save", help="save results", default=True)
    
    parser.add_argument("--horizon", help="horizon in one task", type=int, default=10000)
    parser.add_argument("--repetitions", help="repetitions for configuration", type=int, default=1)
    parser.add_argument("--njobs", help="n_jobs for configuration", type=int, default=40)
    
    parser.add_argument("--a", help="hyperparameter a", type=int, default=0)
    parser.add_argument("--c", help="hyperparameter c", type=int, default=0.5)
    
    args = parser.parse_args()

    main(args)
