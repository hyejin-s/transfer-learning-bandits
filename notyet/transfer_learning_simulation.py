import os
import numpy as np
import argparse
from tqdm import tqdm
from math import ceil

import matplotlib.pyplot as plt

from OSSB_transfer_hyp10 import OSSB_DEL, LipschitzOSSB_DEL, LipschitzOSSB_DEL_true
from policies.Bernolli import Bernoulli
# from environments.evaluator import *
from evaluator_transfer_learning import *
import pickle


def estimate_Lipschitz_constant(thetas, embeddings):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i+1]-thetas[i])/(embeddings[i+1]-embeddings[i]))
    return np.amax(L_values)

def Lipschitz_beta(LC_list, beta, epsilon, M):
    bound_value = ceil(beta * M) # ROUND UP
    L_beta = sorted(LC_list, reverse=True)[bound_value-1] + epsilon
    return L_beta

def plot_transfer(filepath):
    plt.figure()
    X = list(range(1, M+1))
    for i in range(len(POLICIES)):
        plt.plot(X, regret_transfer[i], label="{}".format(labels[i]), color=colors[i])
    plt.legend()
    plt.title("Total {} episode, {} horizon".format(M, args.horizon))
    plt.savefig(filepath+'/Regret', dpi=300)

def main(args):

    print(f"Transfer Learning for Lipschitz Constant {args.L}, the horizon of each instance {args.horizon} ")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    M = args.num_episodes # the ordering of Transfer Learning
    embeddings, num_arms = args.embeddings, args.num_arms
    beta, epsilon = args.beta_hyp[0], args.beta_hyp[1]

    with open(f'./utils/bandit_instances/num_arms_{num_arms}_num_episodes_400_Lipschitz_{args.L}.pickle', 'rb') as f:
        episode_instances = pickle.load(f)

    # for results ------
    true_LC_episode, beta_LC_episode, max_LC_episode = np.zeros(M), np.zeros(M), np.zeros(M)
    regret_episode, empirical_LC_episode = np.zeros(M), np.zeros(M)
    if args.scratch:
        regret_episode, empirical_LC_episode = np.zeros((5, M)), np.zeros((5, M))
    
    # transfer learning -----------------
    for m in range(M):
        # import pdb; pdb.set_trace()
        print("{} episode".format(m+1))

        instance = episode_instances[m]
        print(instance)
        # import pdb; pdb.set_trace()

        ENVIRONMENTS = [{"arm_type": Bernoulli, "params": instance}]
        POLICIES = []
    
        # various Lipschitz Constant -----------------
        true_LC = estimate_Lipschitz_constant(instance, embeddings)
        true_LC_episode[m] = true_LC

        ''' beta '''
        beta_LC = Lipschitz_beta(true_LC_episode[:m+1], beta, epsilon, m)
        beta_LC_episode[m] = beta_LC
        POLICIES.append(LipschitzOSSB_DEL_true(nbArms=num_arms, gamma=0.001, L=beta_LC))

        if args.scratch:
            ''' inf and Lipschitz'''
            POLICIES.append({"archtype":OSSB_DEL, "params":{}})
            POLICIES.append({"archtype":LipschitzOSSB_DEL, "params":{}})

            ''' true '''
            POLICIES.append(LipschitzOSSB_DEL_true(nbArms=num_arms, gamma=0.001, L=true_LC))

            ''' max '''
            max_LC = np.max(true_LC_episode[:m+1])
            max_LC_episode[m] = max_LC
            POLICIES.append(LipschitzOSSB_DEL_true(nbArms=num_arms, gamma=0.001, L=max_LC))

        configuration = {"horizon": args.horizon , "repetitions": args.repetitions, "n_jobs": args.njobs, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
        evaluation = Evaluator(configuration)
    
        for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
            evaluation.startOneEnv(envId, env)
        
        regret_episode[m] = evaluation.getCumulatedRegret_LessAccurate(policyId=0)[args.horizon-1]
        empirical_LC_episode[m] = estimate_Lipschitz_constant(evaluation.get_lastmeans()[0][0], embeddings)

        if args.scratch:
            for idx in range(len(POLICIES)):
                regret_episode[idx][m] = evaluation.getCumulatedRegret_LessAccurate(policyId=idx)[args.horizon-1]
                empirical_LC_episode[idx][m] = estimate_Lipschitz_constant(evaluation.get_lastmeans()[idx][0])

        if args.save:
            with open(f'{args.save_dir}/true_LC_beta_{beta}_epsilon_{epsilon}_L_{args.L}_episodes_{M}.pickle', 'wb') as f:
                pickle.dump(true_LC_episode, f)
            with open(f'{args.save_dir}/beta_LC_beta_{beta}_epsilon_{epsilon}_L_{args.L}_episodes_{M}.pickle', 'wb') as f:
                pickle.dump(beta_LC_episode, f)
            with open(f'{args.save_dir}/max_LC_beta_{beta}_epsilon_{epsilon}_L_{args.L}_episodes_{M}.pickle', 'wb') as f:
                pickle.dump(true_LC_episode, f)
            with open(f'{args.save_dir}/regret_beta_{beta}_epsilon_{epsilon}_L_{args.L}_episodes_{M}.pickle', 'wb') as f:
                pickle.dump(regret_episode, f)
            with open(f'{args.save_dir}/empirical_LC_beta_{beta}_epsilon_{epsilon}_L_{args.L}_episodes_{M}.pickle', 'wb') as f:
                pickle.dump(empirical_LC_episode, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", help="save direction", type=str, default="./results")
    parser.add_argument("--num_arms", type=int, default=6, help="The number of arms")
    parser.add_argument("--num_episodes", help="M; the total number of transfer learning tasks", type=int, default=400)
    parser.add_argument("--L", type=float, default=0.5, help="Lipschitz Constant")
    parser.add_argument("--embeddings", help="The embedding of a bandit instance", nargs='+', type=float, default=[0, 0.8, 0.85, 0.9, 0.95, 1])
    parser.add_argument("--beta_hyp", nargs="+", default=[0.1, 0.05], help="[\beta, \epsilon_{\beta}]; Hyperparamets for ")
    parser.add_argument("--scratch", help="from scratch (true, inf, max)", default=False)
    parser.add_argument("--save", help="save results", default=True)
    
    parser.add_argument("--horizon", help="horizon in one task", type=int, default=10000)
    parser.add_argument("--repetitions", help="repetitions for configuration", type=int, default=1)
    parser.add_argument("--njobs", help="n_jobs for configuration", type=int, default=40)
    
    args = parser.parse_args()

    main(args)