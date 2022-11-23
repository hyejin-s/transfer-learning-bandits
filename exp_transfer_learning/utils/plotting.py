import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

def plotting(value, length, lw, labels, save_path, save_name):
    colors = ['limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple', 'red']
    plt.figure(figsize=(24,16))
    X = list(range(1, length+1))
    if len(value) == length: # draw only one line
        plt.plot(X, value, color=colors[0], linewidth=lw)
    else:
        assert len(value) <= len(colors), 'need more color list'
        assert len(labels) == len(value), 'check labels list'
        for i in range(len(value)):
            plt.plot(X, value[i], label="{}".format(labels[i]), color=colors[i], linewidth=lw)
        plt.legend(fontsize=30)

    plt.xlabel('Episode', fontsize=60)
    plt.ylabel('Lipschitz Constant', fontsize=60)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40) 

    save_dir = os.path.join(save_path, save_name)
    plt.savefig(save_dir+".png", dpi=300)
    plt.savefig(save_dir+".pdf", format='pdf', dpi=300) 
    
def cumul_plotting(value, length, lw, labels, save_path, save_name):
    colors = ['tomato', 'limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple', 'red']
    plt.figure(figsize=(24,16))
    X = list(range(1, length+1))
    if len(value) == length: # draw only one line
        plt.plot(X, np.cumsum(value), color=colors[i], linewidth=lw)
    else:
        assert len(value) <= len(colors), 'need more color list'
        assert len(labels) == len(value), 'check labels list'
        for i in range(len(value)):
            plt.plot(X, np.cumsum(value[i]), label="{}".format(labels[i]), color=colors[i], linewidth=lw)
        plt.legend(fontsize=30)
        
    plt.xlabel('Episode', fontsize=60)
    plt.ylabel('Cumulative Regret', fontsize=60)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40) 

    save_dir = os.path.join(save_path, save_name)
    plt.savefig(save_dir+".png", dpi=300)
    plt.savefig(save_dir+".pdf", format='pdf', dpi=300) 

def main(args):
    print(f"Plotting per episodes for Lipschitz Constant {args.L}")
    
    M, T = args.num_episodes, args.horizon # the ordering of Transfer Learning
    embeddings = []
    exp = args.exp

    # embeddings, num_arms = args.embeddings, len(args.embeddings)
    num_arms = 16
    beta, epsilon = args.beta_hyp[0], args.beta_hyp[1]
    
    eps_beta = args.L * 0.1
    hyp_list = [(0.1, eps_beta), (0.3, eps_beta), (0.5, eps_beta)]

    ''' cumulative regret per episodes'''
    if args.cumregret:
        print(f"Cumulative regret per episodes {M}")
        labels = [r'$\pi(\infty)$', r'$\pi(L_{true})$', r'$\pi(L_{max})$',
                  r'$\pi(\hat{L}_{\beta}): (\beta, \varepsilon_\beta)=$'+f'{hyp_list[0]}', 
                  r'$\pi(\hat{L}_{\beta}): (\beta, \varepsilon_\beta)=$'+f'{hyp_list[1]}', 
                  r'$\pi(\hat{L}_{\beta}): (\beta, \varepsilon_\beta)=$'+f'{hyp_list[2]}', 
                  ]#, 'hyperparameter tuning']
        # labels = ['inf', 'true']#, 'max', f'beta {hyp_list[1]}']#, 'hyperparameter tuning']
        # labels = ['inf', 'true', 'max', f'beta {hyp_list[1]}']#, 'hyperparameter tuning']
        with open(f'{args.save_dir}/exp_{exp}/transfer_learning/regret__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'rb') as f:
            regret_episode = pickle.load(f)
        regret_episode = regret_episode.tolist()

        # with open(f'{args.save_dir}/exp_{exp}/transfer_learning/regret_hyp__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'rb') as f:
        #     hyp_regret = pickle.load(f)
        # regret_episode.append(hyp_regret)
        if not os.path.exists(os.path.join(args.save_dir, f'exp_{exp}/plot')):
            os.makedirs(os.path.join(args.save_dir, f'exp_{exp}/plot'))
        
        save_path = os.path.join(args.save_dir, f'exp_{exp}/plot')
        save_name = f'cumulative_regret_L__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_'

        cumul_plotting(regret_episode, M, args.lw, labels, save_path, save_name)
        print("Success to save plot for cumulative regret per episodes")

    if args.lipschitz:
        print(f"Lipshchitz Constant per episodes {M}")
        labels = [r'$L_{true}$', r'$L_{max}$',
                  r'$\hat{L}_{\beta}: (\beta, \varepsilon_\beta)=$'+f'{hyp_list[0]}', 
                  r'$\hat{L}_{\beta}: (\beta, \varepsilon_\beta)=$'+f'{hyp_list[1]}', 
                  r'$\hat{L}_{\beta}: (\beta, \varepsilon_\beta)=$'+f'{hyp_list[2]}', 
                'hyperparameter tuning']#, 'empirical']
        labels = ['empirical']
        # labels = ['true', 'max', f'beta {hyp_list[0]}', f'beta {hyp_list[1]}', f'beta {hyp_list[2]}', 'empirical']

        # labels = ['true', 'max', f'beta {hyp_list[1]}', 'hyperparameter tuning', 'empirical']
        # labels = ['true', 'max', f'beta {hyp_list[1]}', 'empirical']

        Lipschitz_list = list()
        # with open(f'{args.save_dir}/exp_{exp}/transfer_learning/true_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'rb') as f:
        #     true_LC_episodes = pickle.load(f)
        # print(true_LC_episodes)
        # true_LC_episodes = []
        # for i in range(M):
        #     true_LC_episodes.append(0.02)
        # Lipschitz_list.append(true_LC_episodes)
        
        # with open(f'{args.save_dir}/exp_{exp}/transfer_learning/max_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'rb') as f:
        #     max_LC_episodes = pickle.load(f)
        # Lipschitz_list.append(max_LC_episodes)

        # with open(f'{args.save_dir}/exp_{exp}/transfer_learning/beta_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'rb') as f:
        #     beta_LC_episodes = pickle.load(f)
        # for i in range(len(beta_LC_episodes)):
        #     Lipschitz_list.append(beta_LC_episodes[i])
            
        # with open(f'{args.save_dir}/exp_{exp}/transfer_learning/hyp_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_repe_{args.repetitions}_.pickle', 'rb') as f:
        #     hyp_LC_episodes = pickle.load(f)
        # Lipschitz_list.append(hyp_LC_episodes)
        
        with open(f'{args.save_dir}/exp_{exp}/transfer_learning/empirical_LC__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_.pickle', 'rb') as f:
            empirical_LC_episodes = pickle.load(f)
        Lipschitz_list.append(empirical_LC_episodes)
        
        save_path = os.path.join(args.save_dir, f'exp_{exp}/plot')
        save_name = f'Lipschitz_constance__arms_{num_arms}_episodes_{M}_y_{args.y}_dist_{args.dist}_L_{args.L}_T_{T}_'
        
        plotting(Lipschitz_list, M, args.lw, labels, save_path, save_name)
        print("Success to save plot for Lipschitz Constant per episodes")

    if args.hyp_tuning:
        a, c = 0, 0.5
        save_path = os.path.join(args.save_dir, 'plot')

        if not os.path.exists(os.path.join(args.save_dir, 'plot')):
            os.makedirs(os.path.join(args.save_dir, 'plot'))

        # parameters
        with open(f'{args.save_dir}/hyp_tuning/beta_episode_a_{a}_c_{c}_L_{args.L}_episodes_{M}_.pickle', 'rb') as f:
            beta_LC_episodes = pickle.load(f)
        plotting(beta_LC_episodes, M, args.lw, None, save_path, f'hyp_beta_L_{args.L}_num_episodes_{args.num_episodes}.png')

        with open(f'{args.save_dir}/hyp_tuning/epsilon_episode_a_{a}_c_{c}_L_{args.L}_episodes_{M}_.pickle', 'rb') as f:
            epsilon_LC_episodes = pickle.load(f)
        plotting(epsilon_LC_episodes, M, args.lw, None, save_path, f'hyp_epsilon_L_{args.L}_num_episodes_{args.num_episodes}.png')

        with open(f'{args.save_dir}/hyp_tuning/gamma_episode_a_{a}_c_{c}_L_{args.L}_episodes_{M}_.pickle', 'rb') as f:
            gamma_LC_episodes = pickle.load(f)
        plotting(gamma_LC_episodes, M, args.lw, None, save_path, f'hyp_gamma_L_{args.L}_num_episodes_{args.num_episodes}.png')
        
        # regret
        labels = ['inf', 'Lipschitz', 'true', 'beta (0.1/0.05)', 'max', 'hyp_tuning']
        Lipschitz_list = list()

        with open(f'{args.save_dir}/transfer_learning/regret_beta_{beta}_epsilon_{epsilon}_L_{args.L}_episodes_{M}_.pickle', 'rb') as f:
            regret_episode = pickle.load(f)
        for i in range(len(regret_episode)):
            Lipschitz_list.append(regret_episode[i])

        with open(f'{args.save_dir}/hyp_tuning/regret_LC_hyp_tuning_L_{args.L}_episodes_{M}_.pickle', 'rb') as f:
            regret_hyp = pickle.load(f)
        Lipschitz_list.append(regret_hyp)

        save_path = os.path.join(args.save_dir, 'plot2')
        save_name = f'hyp_cumulative_regret_L_{args.L}_num_episodes_{args.num_episodes}.png'

        cumul_plotting(Lipschitz_list, M, args.lw, labels, save_path, save_name)
        print("Success to save plot for hyp per episodes")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", help="save direction", type=str, default="./results")
    parser.add_argument("--num_episodes", help="M; the total number of transfer learning tasks", type=int, default=400)
    parser.add_argument("--L", help="Lipschitz Constant", type=float, default=0.5)
    parser.add_argument("--dist", type=float, default=0.1)
    parser.add_argument("--y", type=float, default=0.5)
    
    parser.add_argument("--embeddings", help="The embedding of a bandit instance", nargs='+', type=float, default=[0, 0.025, 0.05, 0.075, 0.1, 0.9, 0.925, 0.95, 0.975, 1])
    # parser.add_argument("--embeddings", help="The embedding of a bandit instance", nargs='+', type=float, default=[0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1])
    # parser.add_argument("--embeddings", help="The embedding of a bandit instance", nargs='+', type=float, default=[0, 0.001, 0.999, 1])
    parser.add_argument("--beta_hyp", help="[\beta, \epsilon_{\beta}]", nargs="+", default=[0.1, 0.05])
    parser.add_argument("--horizon", help="horizon in one task", type=int, default=10000)
    parser.add_argument("--repetitions", help="repetitions for configuration", type=int, default=1)
    parser.add_argument("--exp", default=1)
    parser.add_argument("--cumregret", help="plot cumulative regret per episodes", default=False)
    parser.add_argument("--lipschitz", help="plot Lipschitz Constance per episodes", default=True)
    parser.add_argument("--hyp_tuning", help="plot hyperparameter tuning per episodes", default=False)
    
    parser.add_argument("--lw", help="line width", default=5)
    args = parser.parse_args()

    main(args)