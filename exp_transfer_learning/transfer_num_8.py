import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil, sqrt

from sklearn.model_selection import KFold
from scipy.optimize import curve_fit

from policies.Bernolli import Bernoulli
from environments.evaluator_transfer_learning import *
from OSSB_mono_10 import OSSB_DEL, LipschitzOSSB_DEL_true

def zero_division(x):
    return np.max([x, 1e-4])

def Lipschitz_beta(LC_list, beta, epsilon, M):
    bound_value = ceil(beta * M) # ROUND UP
    L_beta = sorted(LC_list, reverse=True)[bound_value-1] + epsilon
    return L_beta

def com_Lipschitz_constant(thetas, embeddings):
    L_values = []
    for i in range(thetas.size-1):
        L_values.append(abs(thetas[i+1]-thetas[i])/(np.linalg.norm(embeddings[i+1]-embeddings[i])))

    return np.amax(L_values)

def main(args):
    print(f"Transfer Learning for Lipschitz Constant {args.L}, the horizon of each instance {args.horizon} ")
    exp = args.exp
    if not os.path.exists(os.path.join(args.save_dir, f'exp_{exp}')):
        os.makedirs(os.path.join(args.save_dir, f'exp_{exp}'))
    
    # num_arms = 8    
    embeddings = []
    dist, y = args.dist, args.y
    embeddings = np.array([[0, 0, 0, 0], [dist, 0, 0, 0], [dist/2, 0, sqrt(3) * (dist/2), 0], 
                       [dist/2, 0, dist/(2*sqrt(3)), (dist*sqrt(6))/3],
                       [0, y, 0, 0], [dist, y, 0, 0], [dist/2, y, sqrt(3) * (dist/2), 0], 
                       [dist/2, y, dist/(2*sqrt(3)), (dist*sqrt(6))/3],
                       ])

    num_arms = len(embeddings)
    
    M, T = args.num_episodes, args.horizon # the ordering of Transfer Learning
    # embeddings, num_arms = args.embeddings, len(args.embeddings)
    
    save_path = os.path.join(args.save_dir, f'exp_{exp}/transfer_learning')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ''' bandit instances '''
    with open(f'./bandit_instances/arms_{len(embeddings)}_y_{args.y}_L_{args.L}_{args.num}.pickle', 'rb') as f:
        episode_instances = pickle.load(f)
    
    # for results ------
    ''' Empirical Lipschitz Constant '''
    empirical_LC_episode = np.zeros(M)
    
    ''' regret '''
    regret_episode = np.zeros(M)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # get empirical LC ----------
    for m in range(M):
        print("{} episode".format(m+1))
        
        instance = episode_instances[m]
        # generate model ----------
        ENVIRONMENTS = [{"arm_type": Bernoulli, "params": instance}]
        POLICIES = []
        
        POLICIES.append({"archtype":OSSB_DEL, "params":{}})
        
        # start ----------
        configuration = {"embeddings": embeddings, "horizon": T, "n_jobs": args.njobs, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
        evaluation = Evaluator(configuration)
        for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
            evaluation.startOneEnv(envId, env)
        
        # save results ----------
        ''' regret '''
        regret_episode[m] = evaluation.getCumulatedRegret_MoreAccurate(policyId=0)[T-1]
        empirical_LC_episode[m] = com_Lipschitz_constant(evaluation.get_lastmeans()[0][0], embeddings)
    
    save_path = os.path.join(args.save_dir, f'exp_{exp}/transfer_learning')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if args.save:    
        with open(f'{save_path}/regret__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_.pickle', 'wb') as f:
            pickle.dump(regret_episode, f)
        with open(f'{save_path}/empirical_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_.pickle', 'wb') as f:
            pickle.dump(empirical_LC_episode, f)

    ''' empirical Lipschitz Constance (saving in advance) '''
    with open(f'{save_path}/empirical_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_.pickle', 'rb') as f:
        empirical_LC_episodes = pickle.load(f)
        
    # hyp ----------
    a, c, q = 0, 0.1, 1

    # for results ----------
    beta_episode, epsilon_episode, gamma_episode, beta_LC_episode = np.zeros(M), np.zeros(M), np.zeros(M), np.zeros(M)
    
    for m in range(M):
        # print("{} episode".format(m+1))
        if m == 0:
            pre_estimated_L = empirical_LC_episodes[0]
        else:
            # pre_estimated_L = empirical_LC_episodes[m-1]
            pre_estimated_L = max(empirical_LC_episodes[:m+1])
        ''' hyperparemter tuning '''
        cumulated_LC = empirical_LC_episodes[:m+1]
        max_L= np.max(cumulated_LC)
        
        q_kfold= list()
        if m < args.kfold:
            q_kfold.append(q)
        else:
            rkf = KFold(n_splits=args.kfold)
            for i in rkf.split(cumulated_LC): # kfold
                A = [cumulated_LC[idx] for idx in i[1]]
                max_L_fold = np.max(A)
                q_kfold.append(max_L/max_L_fold)
            q = np.mean(q_kfold)

        # get estimated L
        estimated_L = q * pre_estimated_L

        z = np.linspace(a, estimated_L, args.linspace)
        cumul_xi_LC = [estimated_L - Lm for Lm in cumulated_LC]
        alpha_z = [len(np.where(cumul_xi_LC < i)[0]) / (m+1) for i in z]

        def func(x, gamma):
            return ((x - a)/zero_division(estimated_L - a)) ** gamma
        
        # get gamma using curve fitting and epsilon_beta using c \in (0, 1)
        gamma, _ = curve_fit(func, z, alpha_z)
        epsilon = c * pre_estimated_L
        a = c * epsilon
        
        ### if estimated_L is small, too small beta
        if gamma > 1:
    
            beta = np.mean(beta_episode[0:m])
        else:
            beta = ((epsilon - a)/zero_division(estimated_L - a)) ** gamma
        beta = 0.01 if beta < 0.01 else 1 if beta >= 1 else beta
            
        # for saving
        beta_episode[m] = beta
        epsilon_episode[m] = epsilon
        gamma_episode[m] = gamma
        print(beta, epsilon, gamma)
        
        # transfer learning ----------
        instance = episode_instances[m]

        ''' beta '''
        beta_LC = Lipschitz_beta(cumulated_LC, beta, epsilon, m)
        beta_LC_episode[m] = beta_LC    
    
    save_dir = os.path.join(args.save_dir, f'exp{exp}/hyp_tuning')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.save:
        with open(f'{args.save_dir}/exp{exp}/hyp_tuning/beta_episode__a_{args.a}_c_{args.c}_L_{args.L}_episodes_{M}_T_{T}_.pickle', 'wb') as f:
            pickle.dump(beta_episode, f)
        with open(f'{args.save_dir}/exp{exp}/hyp_tuning/epsilon_episode__a_{args.a}_c_{args.c}_L_{args.L}_episodes_{M}_T_{T}_.pickle', 'wb') as f:
            pickle.dump(epsilon_episode, f)
        with open(f'{args.save_dir}/exp{exp}/hyp_tuning/gamma_episode__a_{args.a}_c_{args.c}_L_{args.L}_episodes_{M}_T_{T}_.pickle', 'wb') as f:
            pickle.dump(gamma_episode, f)

    ''' empirical Lipschitz Constance (saving in advance) '''
    with open(f'{save_path}/empirical_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_.pickle', 'rb') as f:
        empirical_LC_episodes = pickle.load(f)

    eps_beta = args.L * 0.1
    hyp_list = [[0.1, eps_beta], [0.3, eps_beta], [0.5, eps_beta]]
    
    ''' hyper-parameters tuning value '''
    with open(f'{args.save_dir}/exp{exp}/hyp_tuning/beta_episode__a_{args.a}_c_{args.c}_L_{args.L}_episodes_{M}_T_{T}_.pickle', 'rb') as f:
        beta_episode = pickle.load(f)
    with open(f'{args.save_dir}/exp{exp}/hyp_tuning/epsilon_episode__a_{args.a}_c_{args.c}_L_{args.L}_episodes_{M}_T_{T}_.pickle', 'rb') as f:
        epsilon_episode = pickle.load(f)
    
    # for results ------
    ''' empirical Lipschitz constant '''
    true_LC_episode, max_LC_episode = np.zeros(M), np.zeros(M)
    beta_LC_episode = np.zeros((len(hyp_list), M))
    beta_hyp_LC_episode = np.zeros(M)
    
    ''' regret '''
    regret_episode = np.zeros((3+len(hyp_list), M))
    regret_episode_hyp = np.zeros(M)
    
    save_path = os.path.join(args.save_dir, f'exp_{exp}/transfer_learning')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # start transfer learning ----------
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
        POLICIES.append(LipschitzOSSB_DEL_true(nbArms=num_arms, embeddings=embeddings, gamma=0.001, L=true_LC))
        
        ''' max '''
        max_LC = np.max(empirical_LC_episodes[:m+1])     
        max_LC_episode[m] = max_LC
        POLICIES.append(LipschitzOSSB_DEL_true(nbArms=num_arms, embeddings=embeddings, gamma=0.001, L=max_LC))
        
        ''' beta '''
        for idx in range(len(hyp_list)):
            beta_LC = Lipschitz_beta(empirical_LC_episodes[:m+1], hyp_list[idx][0], hyp_list[idx][1], m)
            # print(beta_LC)
            beta_LC_episode[idx][m] = beta_LC
            POLICIES.append(LipschitzOSSB_DEL_true(nbArms=num_arms, embeddings=embeddings, gamma=0.001, L=beta_LC))
            
        ''' hyp '''
        hyp_LC = Lipschitz_beta(empirical_LC_episodes[:m+1], beta_episode[m], epsilon_episode[m], m)
        beta_hyp_LC_episode[m] = hyp_LC
        POLICIES.append(LipschitzOSSB_DEL_true(nbArms=num_arms, embeddings=embeddings, gamma=0.001, L=hyp_LC))

        # start ----------
        configuration = {"embeddings": embeddings, "horizon": args.horizon, "repetitions": args.repetitions, "n_jobs": args.njobs, "verbosity": 40, "environment": ENVIRONMENTS, "policies": POLICIES}
        evaluation = Evaluator(configuration)
        for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
            evaluation.startOneEnv(envId, env)
        
        # save results ----------
        ''' regret '''
        for idx in range(len(POLICIES)-1):
            regret_episode[idx][m] = evaluation.getCumulatedRegret_MoreAccurate(policyId=idx)[args.horizon-1]
        regret_episode_hyp[m] = evaluation.getCumulatedRegret_MoreAccurate(policyId=len(POLICIES)-1)[args.horizon-1]
        
    if args.save:    
        with open(f'{save_path}/regret__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'wb') as f:
            pickle.dump(regret_episode, f)
        with open(f'{save_path}/regret_hyp__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'wb') as f:
            pickle.dump(regret_episode_hyp, f)
            
        with open(f'{save_path}/true_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'wb') as f:
            pickle.dump(true_LC_episode, f)
        with open(f'{save_path}/max_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'wb') as f:
            pickle.dump(max_LC_episode, f)
        with open(f'{save_path}/beta_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'wb') as f:
            pickle.dump(beta_LC_episode, f)
        with open(f'{save_path}/hyp_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'wb') as f:
            pickle.dump(beta_hyp_LC_episode, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", help="save direction", type=str, default="./results")
    parser.add_argument("--num_episodes", help="M; the total number of transfer learning tasks", type=int, default=400)
    parser.add_argument("--L", help="Lipschitz Constant", type=float, default=0.5)
    # parser.add_argument("--embeddings", help="The embedding of a bandit instance", nargs='+', type=float, default=[0, 0.025, 0.05, 0.075, 0.1, 0.9, 0.925, 0.95, 0.975, 1])
    parser.add_argument("--embeddings", help="The embedding of a bandit instance", nargs='+', type=float, default=[0, 0.01, 0.99, 1])
    parser.add_argument("--exp", help="experiment num", default=1)
    parser.add_argument("--num", help="experiment num", default=0)

    parser.add_argument("--save", help="save results", default=True)

    parser.add_argument("--kfold", help="k; episode k-fold", default=5)    
    parser.add_argument("--linspace", help="The number of fraction for np.linspace", type=int, default=50)

    
    parser.add_argument("--horizon", help="horizon in one task", type=int, default=10000)
    parser.add_argument("--repetitions", help="repetitions for configuration", type=int, default=1)
    parser.add_argument("--njobs", help="n_jobs for configuration", type=int, default=40)
    
    parser.add_argument("--a", help="hyperparameter a", type=int, default=0)
    parser.add_argument("--c", help="hyperparameter c", type=int, default=0.5)
    
    parser.add_argument("--dist", type=float, default=0.1)
    parser.add_argument("--y", type=float, default=0.5)

    args = parser.parse_args()

    main(args)
