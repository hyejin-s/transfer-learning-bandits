colors = ['tomato', 'limegreen', 'deepskyblue', 'crimson', 'pink', 'mediumorchid', 'rebeccapurple'] # 7
labels = ['(0.1, 0.05)', '(0.3, 0.05)', '(0.5, 0.05)', 'true', 'max', 'inf']


with open(dir_name+ "/EmpiricalLipschitz.txt", "w") as f:
    for i in range(len(Empirical_Lipschitz)):
        f.write("\n{}".format(i))
        for episode in range(len(Empirical_Lipschitz[i])):
            f.write("\nepisode:{}, {}".format(episode, Empirical_Lipschitz[i, episode]))

with open(dir_name+ "/regret_transfer.txt", "w") as f:
    for episode in range(M): 
        f.write("\nepisode:{}, {}, {}, {}, {}, {}, {}".format(episode, regret_transfer[0, episode], regret_transfer[1, episode], regret_transfer[2, episode], regret_transfer[3, episode], regret_transfer[4, episode], regret_transfer[5, episode]))


#lastpull = np.zeros((M,7,10)) #(M, nbPolicies, numArms)
with open(dir_name+ "/lastpull.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}".format(m))
        for policyId in range(6):
            f.write("\npolicy:{}\n".format(policyId))
            np.savetxt(f, lastpull[m][policyId].astype(int), fmt='%i', newline=", ")

with open(dir_name+ "/lastmean.txt", "w") as f:
    for m in range(M):
        f.write("\nepisode:{}".format(m))
        for policyId in range(6):
            f.write("\npolicy:{}\n".format(policyId))
            np.savetxt(f, lastmean[m][policyId], newline=", ", fmt='%1.3f')

with open(dir_name+ "/information.txt", "w") as f:
    f.write("Number of episodes: " + str(M))
    f.write("\nHorizon: " + str(args.horizon))
    f.write("\n(beta, epsilon): " + str(valuelist))
    f.write("\nembeddings: "+str(embeddings))

plotRegret(dir_name)

# print(len(episode_instance))
    # [beta, epsilon] # args.beta_hyp

    # valuelist = [[0.1, 0.05], [0.3, 0.05], [0.5, 0.05]] # [beta, epsilon]

    # with open(r"/home/phj/TransferLearning/TransferSimulation/ha01/EmpiricalLipschitz.txt") as f:
    #     data = f.read().split()
    #     for i in range(len(valuelist)+1):
    #         Lip_list = []
    #         Empirical_list = []
    #         if i == 3:
    #             for m in range(M):
    #                 Empirical_list.append(float(data[m*2+1]))
    #                 L_info[i][m] = max(Empirical_list)
    #         else:
    #             beta = valuelist[i][0]
    #             epsilon = valuelist[i][1]
    #             for m in range(M):
    #                 Empirical_list.append(float(data[m*2+1]))
    #                 L_info[i][m] = Lipschitz_beta(Empirical_list, epsilon, beta, m)

    # arm_info = np.zeros((M,6))
    # with open(r"/home/phj/TransferLearning/TransferSimulation/ha01/arm_info.txt") as f:
    #     data = f.read().split()
    #     for m in range(M):
    #         for idx in range(6):
    #             if idx == 0:
    #                 arm_info[m][idx] = data[7*m+1+idx][1:len(data[7*m+1+idx])]
    #             elif idx == 5:
    #                 arm_info[m][idx] = data[7*m+1+idx][:-1]
    #             else:
    #                 arm_info[m][idx] = data[7*m+1+idx]

    # regret_transfer = np.zeros((6,M))

    # for print
    # lastpull = np.zeros((M,6,6)) #(M, nbPolicies, numArms)
    # lastmean = np.zeros((M,6,6))

    ##### Transfer Learning #####
    # Empirical_Lipschitz = np.zeros((6,M))   # store empirical Lipschitz constant every episode
    # # POLICIES = [{"archtype":OSSB_DEL, "params":{}}, {"archtype":LipschitzOSSB_DEL, "params":{}}]   