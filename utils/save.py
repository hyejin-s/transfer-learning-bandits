def savedata(self, filepath, envId=0, moreAccurate=True):
        # information (arms, embeddings, repetitions, epsilon, GAMMA)
        with open(filepath+ "/information.txt", "w") as f:
            f.write("arms:\n")
            np.savetxt(f, arms, fmt="%.4f", newline=", ")
            f.write("\n")
            f.write("embeddings:\n")
            np.savetxt(f, embeddings, fmt="%.4f", newline=", ")
            f.write("\n")
            f.write("repetitions: " + str(self.repetitions) + "\n")
            f.write("GAMMA: " + str(GAMMA) + "\n")
            f.write("EPSILON: " + str(EPSILON) + "\n")

        # pulls data
        """
        with open(filepath+ "/Pulls.txt", "w") as f:
            for i, policyId in enumerate(self.policies):
                f.write('\nPolicy:{}'.format(policyId.__cachedstr__))
                for repeatId in range(self.repetitions):)
                    f.write("\nrepeat:{}".format(repeatId))
                    for t in range(self.horizon):
                        if t % 100 == 0:
                            f.write("\ntime:{}, ".format(t+1))
                            np.savetxt(f, self.allPulls_rep[envId][repeatId, i, :, t].astype(int), fmt='%i', newline=", ")
                        # delimiter=", "
        """
        
        # number of arm pulls
        """
        with open(filepath+ "/num_ArmPulls.txt", "w") as f:
            cum_pullarm = np.zeros((self.nbPolicies, self.repetitions, self.envs[envId].nbArms))
            for i, policyId in enumerate(self.policies):
                f.write('\nPolicy:{}'.format(policyId.__cachedstr__))
                for repeatId in range(self.repetitions):
                    f.write('\nRepeat: '+str(repeatId))
                    for t in range(self.horizon):
                        cum_pullarm[i][repeatId] += self.allPulls_rep[envId][repeatId, i, :, t]
                        f.write("\n")
                        np.savetxt(f, cum_pullarm[i][repeatId].astype(int), fmt='%i', newline=", ")
        """

        with open(filepath+ "/num_ArmPulls_last.txt", "w") as f:
            cum_pullarm = np.zeros((self.nbPolicies, self.repetitions, self.envs[envId].nbArms))
            for i, policyId in enumerate(self.policies):
                f.write('\nPolicy:{}'.format(policyId.__cachedstr__))
                for repeatId in range(self.repetitions):
                    f.write('\nRepeat: {}, '.format(repeatId))
                    for t in range(self.horizon):
                        cum_pullarm[i][repeatId] += self.allPulls_rep[envId][repeatId, i, :, t]
                    np.savetxt(f, cum_pullarm[i][repeatId].astype(int), fmt='%i', newline=", ")

            """
            for i, policyId in enumerate(self.policies):
                f.write('\nPolicy:{}'.format(policyId.__cachedstr__))
                last_pull = np.transpose(self.lastPulls[envId][i]) # (repetition, arm)
                for repeatId in range(self.repetitions):
                    f.write("\n")
                    f.write('Repeat: '+str(repeatId))
                    f.write("\n")
                    np.savetxt(f, last_pull[repeatId].astype(int), fmt='%i', newline=", ")
            """

        # estimated Lipschitz Constant
        with open(filepath+ "/estimated_Lipschitz_Constant.txt", "w") as f:
            for repeatId in range(self.repetitions):
                f.write('\nRepeat: '+str(repeatId))
                for t in range(self.horizon):
                    #if t % 1000 == 1:
                    f.write('\nt={}, '.format(t))
                    f.write(str(self.estimatedLipschitz[envId][repeatId][t]))
            
        with open(filepath+ "/estimated_Lipschitz_Constant_last.txt", "w") as f:
            for repeatId in range(self.repetitions):
                f.write("\n")
                f.write('Repeat: '+str(repeatId))
                f.write("\n")
                f.write(str(self.estimatedLipschitz[envId][repeatId][self.horizon-1]))
                f.write("\n")
        
        # Result of Linear Programming
        with open(filepath+"/LP_solutions.txt", "w") as f:
            for i, policy in enumerate(self.policies):
                f.write("\nPolicy: " + str(policy) + "\n")
                for repeatId in range(self.repetitions):
                    f.write("\nRepeat: "+str(repeatId))
                    f.write("\n")
                    for t in range(self.horizon):
                        #if t % 1000 == 1:
                        f.write("\nt={}, ".format(t))
                        np.savetxt(f, self.etaSolution[envId][i][repeatId][t], newline=", ", fmt='%.4f')
        
        with open(filepath+"/LPcompare_solutions.txt", "w") as f:
            for i, policy in enumerate(self.policies):
                f.write("\nPolicy: " + str(policy) + "\n")
                for repeatId in range(self.repetitions):
                    f.write("\nRepeat: "+str(repeatId))
                    f.write("\n")
                    for t in range(self.horizon):
                        #if t % 100 == 1:
                        f.write("\nt={}, ".format(t))
                        np.savetxt(f, self.etaCompare[envId][i][repeatId][t], newline=", ", fmt='%.4f')

        with open(filepath+"/LPcompare_solutions_last.txt", "w") as f:
            for i, policy in enumerate(self.policies):
                f.write("\nPolicy: " + str(policy) + "\n")
                for repeatId in range(self.repetitions):
                    f.write("\nRepeat: "+str(repeatId))
                    f.write("\n")
                    np.savetxt(f, self.etaCompare[envId][i][repeatId][self.horizon-1], newline=", ", fmt='%.4f')

        with open(filepath+"/action_compareInfo.txt", "w") as f:
            f.write("[estimation, exploitation, exploration]")
            for i, policy in enumerate(self.policies):
                f.write("\nPolicy: " + str(policy) + "\n")
                for repeatId in range(self.repetitions):
                    f.write("\nRepeat: "+str(repeatId))
                    f.write("\n")
                    for t in range(self.horizon):
                        #if t % 100 == 1:
                        f.write("\nt={}, ".format(t))
                        np.savetxt(f, self.compareInfo[envId][i][repeatId][t], newline=", ", fmt='%.4f')
         
        # define cumreward and cumpull
        with open(filepath+ "/arms_cumreward.txt", "w") as f:
            for i, policy in enumerate(self.policies):
                f.write('\nPolicy:{}'.format(policy.__cachedstr__))
                for repeatId in range(self.repetitions):
                    f.write('\nRepeat: '+str(repeatId))
                    for t in range(self.horizon):
                        #if t % 100 == 1:
                        f.write("\nt= "+str(t)+", ")
                        np.savetxt(f, self.cumreward[envId][i][repeatId, :, t].astype(int), fmt='%i', newline=", ")
        
        with open(filepath+ "/arms_cumpull.txt", "w") as f:
            for i, policy in enumerate(self.policies):
                f.write('\nPolicy:{}'.format(policy.__cachedstr__))
                for repeatId in range(self.repetitions):
                    f.write('\nRepeat: '+str(repeatId))
                    for t in range(self.horizon):
                        #if t % 100 == 1:
                        f.write("\nt= "+str(t)+", ")
                        np.savetxt(f, self.cumpull[envId][i][repeatId, :, t].astype(int), fmt='%i', newline=", ")
    

        with open(filepath+ "/Empirical_means.txt", "w") as f:
            for i, policy in enumerate(self.policies):
                f.write('\nPolicy:{}'.format(policy.__cachedstr__))
                for repeatId in range(self.repetitions):
                    f.write('\nRepeat: '+str(repeatId))
                    for t in range(self.horizon):
                        if 0 in self.cumpull[envId][i][repeatId, :, t]: # for divide                         
                            self.cumpull[envId][i][repeatId, :, t] = np.where(self.cumpull[envId][i][repeatId, :, t]==0, 1, self.cumpull[envId][i][repeatId, :, t])
                        #if t % 100 == 1:
                        f.write("\nt= "+str(t)+", ")
                        mean_divide = self.cumreward[envId][i][repeatId, :, t]/self.cumpull[envId][i][repeatId, :, t]
                        np.savetxt(f, mean_divide, fmt='%1.3f', newline=", ")

        # def get_lastmeans()
        with open(filepath+ "/Empirical_means_last.txt", "w") as f:
            for i, policy in enumerate(self.policies):
                f.write('\nPolicy:{}'.format(policy.__cachedstr__))
                for repeatId in range(self.repetitions):
                    f.write('\nRepeat: '+str(repeatId))
                    f.write("\n")
                    np.savetxt(f, self.get_lastmeans()[i][repeatId], fmt='%1.3f', newline=", ")
    

def plotHistogram(self, envId=0, horizon=10000, savefig=None, bins=50, alpha=0.9, density=None):
    """Plot a horizon=10000 draws of each arms."""
    arms = self.arms
    rewards = np.zeros((len(arms), horizon))
    colors = palette(len(arms))
    for armId, arm in enumerate(arms):
        if hasattr(arm, 'draw_nparray'):  # XXX Use this method to speed up computation
            rewards[armId] = arm.draw_nparray((horizon,))
        else:  # Slower
            for t in range(horizon):
                rewards[armId, t] = arm.draw(t)
    # Now plot
    fig = plt.figure()
    for armId, arm in enumerate(arms):
        plt.hist(rewards[armId, :], bins=bins, density=density, color=colors[armId], label='$%s$' % repr(arm), alpha=alpha)
    legend()
    plt.xlabel("Rewards")
    if density:
        plt.ylabel("Empirical density of the rewards")
    else:
        plt.ylabel("Empirical count of observations of the rewards")
    plt.title("{} draws of rewards from these arms.\n{} arms: {}{}".format(horizon, self.nbArms, self.reprarms(latex=True), signature))
    show_and_save(showplot=True, savefig=savefig, fig=fig, pickleit=False)
    return fig


def plotArmPullsBar(self, envId=0):
    fig = plt.figure()
    colors = palette(self.nbPolicies)
    markers = makemarkers(self.nbPolicies)
    numpolicy = self.nbPolicies
    for i, policy in enumerate(self.policies):
        X = np.array([numpolicy*value + 0.8*i for value in range(self.envs[envId].nbArms)])
        Y = self.pulls[envId][i] / float(self.repetitions)
        plt.bar(X, Y, label=policy.__cachedstr__, color=colors[i])
    X = np.arange(self.envs[envId].nbArms)
    plt.xticks([i*numpolicy + 2*(1/numpolicy) for i in range(self.envs[envId].nbArms)] ,X)
    legend()
    plt.xlabel("Order of Arms")
    plt.ylabel("number of draws")
    plt.title("Mean number of times each arm pulls in ${}$ for different bandit algorithms, \naveraged ${}$ times: ${}$ arms".format(self.horizon, self.repetitions, self.envs[envId].nbArms))
    plt.savefig(dir_name+'/num_ArmPulls_bar', dpi=300)
    show_and_save(self.showplot, savefig, fig=fig, pickleit=USE_PICKLE)
    return fig       

def plotArmPulls(self, filepath, envId=0):
    fig = plt.figure()
    colors = palette(self.nbPolicies)
    markers = makemarkers(self.nbPolicies)
    cum_pullarm = np.zeros((self.nbPolicies, self.repetitions, self.envs[envId].nbArms))
    for i, policyId in enumerate(self.policies):
        for repeatId in range(self.repetitions):
            for t in range(self.horizon):
                cum_pullarm[i, repeatId] += self.allPulls_rep[envId][repeatId, i, :, t]
        X = np.arange(self.envs[envId].nbArms)
        meanY = cum_pullarm[i].mean(axis=0)
        std = np.std(cum_pullarm[i], axis=0)/np.sqrt(self.repetitions)
        ci = 1.96*std
        plt.plot(X, meanY, label=policyId.__cachedstr__)
        plt.fill_between(X, meanY-ci, meanY+ci, alpha=0.3)                           
    legend()
    plt.xlabel("Order of Arms")
    plt.ylabel("number of draws")
    plt.title("Mean number of times each arm pulls in ${}$ for different bandit algorithms, \naveraged ${}$ times: ${}$ arms".format(self.horizon, self.repetitions, self.envs[envId].nbArms))
    plt.savefig(filepath+'/num_ArmPulls', dpi=300)
    return fig

# Error: self.lastPulls
def plotArmPulls_percent_test(self, filepath, envId=0):
    fig = plt.figure()
    colors = ['tomato', 'orangered', 'yellowgreen', 'limegreen', 'dodgerblue', 'deepskyblue']
    markers = makemarkers(self.nbPolicies)
    percent = int(self.repetitions // 10) # num of plotting
    X = np.arange(self.envs[envId].nbArms)
    for i, policyId in enumerate(self.policies):
        standard_list = np.transpose(self.lastPulls[envId][i])
        for num in range(self.envs[envId].nbArms):
            sort_down = standard_list[standard_list[:,num].argsort()][0:self.repetitions, num:num+1]
            standard_list[0:self.repetitions, num:(num+1)] = sort_down
        top_num, bottom_num = standard_list, standard_list
        for j in range(self.repetitions-percent-1):
            top_num = np.delete(top_num, 0, 0)
            bottom_num = np.delete(bottom_num, self.repetitions-1-j, 0)
        meantop_num, maxtop_num, mintop_num = top_num.mean(axis=0), top_num.max(axis=0), top_num.min(axis=0)
        plt.plot(X, meantop_num, label=policyId.__cachedstr__+'_top', color = colors[i*2])
        plt.fill_between(X, mintop_num, maxtop_num, alpha=0.3,color = colors[i*2])
        meanbottom_num, maxbottom_num, minbottom_num = bottom_num.mean(axis=0), bottom_num.max(axis=0), bottom_num.min(axis=0)
        plt.plot(X, meanbottom_num, label=policyId.__cachedstr__+'_bottom', color = colors[i*2+1])
        plt.fill_between(X, minbottom_num, maxbottom_num, alpha=0.3, color = colors[i*2+1])
    legend()
    plt.xlabel("Order of Arms")
    plt.ylabel("number of draws")
    plt.title("Top and bottom(10%) number of times each arm pulls in ${}$ for different bandit algorithms, \naveraged ${}$ times: ${}$ arms".format(self.horizon, self.repetitions, self.envs[envId].nbArms))
    plt.savefig(filepath+'/num_ArmPulls_top_bottom10%mean', dpi=300)
    return fig

# check lastPulls
def check_value(self, filepath, envId=0):
    cum_pullarm = np.zeros((self.nbPolicies, self.repetitions, self.envs[envId].nbArms))
    for i, policyId in enumerate(self.policies):
        for repeatId in range(self.repetitions):
            for t in range(self.horizon):
                cum_pullarm[i, repeatId] += self.allPulls_rep[envId][repeatId, i, :, t]

    with open(filepath+"/check_value.txt", "w") as f:
        for i, policy in enumerate(self.policies):
            f.write("\nPolicy: " + str(policy) + "\n")
            for repeatId in range(self.repetitions):
                f.write("\nRepeat: "+str(repeatId))
                f.write("\n")
                np.savetxt(f, cum_pullarm[i][repeatId], newline=", ", fmt='%i')
                f.write("\n")
                np.savetxt(f, self.lastPulls[envId][i,:,repeatId], newline=", ", fmt='%i')

def plotArmPulls_percent(self, filepath, envId=0):
    fig = plt.figure()
    colors = palette(self.nbPolicies)
    #['tomato', 'orangered', 'yellowgreen', 'limegreen', 'dodgerblue', 'deepskyblue']
    markers = makemarkers(self.nbPolicies)
    cum_pullarm = np.zeros((self.nbPolicies, self.repetitions, self.envs[envId].nbArms))
    percent = int(self.repetitions // 10) # num of plotting
    X = np.arange(self.envs[envId].nbArms)
    for i, policyId in enumerate(self.policies):
        for repeatId in range(self.repetitions):
            for t in range(self.horizon):
                cum_pullarm[i, repeatId] += self.allPulls_rep[envId][repeatId, i, :, t]
        standard_list = cum_pullarm[i]
        for num in range(self.envs[envId].nbArms):
            sort_down = standard_list[standard_list[:,num].argsort()][0:self.repetitions, num:num+1]
            standard_list[0:self.repetitions, num:(num+1)] = sort_down
        top_num, bottom_num = standard_list, standard_list
        for j in range(self.repetitions-percent-1):
            top_num = np.delete(top_num, 0, 0)
            bottom_num = np.delete(bottom_num, self.repetitions-1-j, 0)
        meantop_num, maxtop_num, mintop_num = top_num.mean(axis=0), top_num.max(axis=0), top_num.min(axis=0)
        plt.plot(X, meantop_num, label=policyId.__cachedstr__+'_top', color = colors[i])
        meanbottom_num, maxbottom_num, minbottom_num = bottom_num.mean(axis=0), bottom_num.max(axis=0), bottom_num.min(axis=0)
        plt.plot(X, meanbottom_num, label=policyId.__cachedstr__+'_bottom', color = colors[i])
    legend()
    plt.xlabel("Order of Arms")
    plt.ylabel("number of draws")
    plt.title("Top and bottom(10%) number of times each arm pulls in ${}$ for different bandit algorithms, \naveraged ${}$ times: ${}$ arms".format(self.horizon, self.repetitions, self.envs[envId].nbArms))
    plt.savefig(filepath+'/num_ArmPulls_top_bottom10%', dpi=300)
    return fig

def plotArmPulls_percent_mean(self, filepath, envId=0):
    fig = plt.figure()
    colors = palette(self.nbPolicies)
    markers = makemarkers(self.nbPolicies)
    cum_pullarm = np.zeros((self.nbPolicies, self.repetitions, self.envs[envId].nbArms))
    percent = self.repetitions // 10 # num of plotting
    X = np.arange(self.envs[envId].nbArms)
    for i, policyId in enumerate(self.policies):
        for repeatId in range(self.repetitions):
            for t in range(self.horizon):
                cum_pullarm[i, repeatId] += self.allPulls_rep[envId][repeatId, i, :, t]
        standard_list = cum_pullarm[i]
        for num in range(self.envs[envId].nbArms):
            sort_down = standard_list[standard_list[:,num].argsort()][0:self.repetitions, num:num+1]
            standard_list[0:self.repetitions, num:(num+1)] = sort_down
        top_num, bottom_num = standard_list, standard_list
        for j in range(self.repetitions-percent-1):
            top_num = np.delete(top_num, 0, 0)
            bottom_num = np.delete(bottom_num, self.repetitions-1-j, 0)
        meantop_num, maxtop_num, mintop_num = top_num.mean(axis=0), top_num.max(axis=0), top_num.min(axis=0)
        plt.plot(X, meantop_num, label=policyId.__cachedstr__+'_top', color = colors[i])
        plt.fill_between(X, mintop_num, maxtop_num, alpha=0.3)
        meanbottom_num, maxbottom_num, minbottom_num = bottom_num.mean(axis=0), bottom_num.max(axis=0), bottom_num.min(axis=0)
        plt.plot(X, meanbottom_num, label=policyId.__cachedstr__+'_bottom', color = colors[i])
        plt.fill_between(X, minbottom_num, maxbottom_num, alpha=0.3)
    legend()
    plt.xlabel("Order of Arms")
    plt.ylabel("number of draws")
    plt.title("Top and bottom(10%) number of times each arm pulls in ${}$ for different bandit algorithms, \naveraged ${}$ times: ${}$ arms".format(self.horizon, self.repetitions, self.envs[envId].nbArms))
    plt.savefig(filepath+'/num_ArmPulls_top_bottom10%mean', dpi=300)
    return fig

def plotLipschitz_hist(self, filepath, envId=0):
    fig = plt.figure()
    colors = ['tomato']
    Lips_list = np.zeros((self.repetitions))
    
    for repeatId in range(self.repetitions):
        Lips_value = self.estimatedLipschitz[envId][repeatId][self.horizon-1]
        Lips_list[repeatId] = Lips_value
    plt.hist(Lips_list, label=self.policies[1].__cachedstr__, color=colors[0])
    legend()
    plt.xlabel('Lipschitz Constant')
    plt.ylabel('Number of repetition: total {}'.format(self.repetitions))
    plt.title("Lipschitz Constant Histogram")
    plt.savefig(filepath+'/Lipschitz_Constant_Histogram', dpi=300)
    return fig 

# not yet
# Error: self.lastPulls
def plotLipschitz_restrict(self,filepath,envId=0):
    fig = plt.figure()
    X = np.arange(self.envs[envId].nbArms)
    Y_list = []
    repeat_list = []

    last_pull = np.transpose(self.lastPulls[envId][1]) # repetition, arm
    for repeatId in range(self.repetitions):
        if self.estimatedLipschitz[envId][repeatId][self.horizon-1] < ref_value:
            Y = last_pull[repeatId]
            Y_list.append(Y)
            repeat_list.append(repeatId)
            plt.plot(X, Y, label=self.policies[1].__cachedstr__+", constant: "+str(self.estimatedLipschitz[envId][repeatId][self.horizon-1])+", repeatId: "+str(repeatId))
    #legend()
    #plt.xlabel('Ordering of arms')
    #plt.ylabel('Number of arms pulls')
    #plt.title("Lipschitz Constant Histogram")
    #plt.savefig(filepath+'/Lipschitz_Constant_restrict', dpi=300)

    with open(filepath+ "/Condition_repeatId.txt", "w") as f:
        for i in range(len(repeat_list)):
            f.write("\n"+str(repeat_list[i])+": ")
            f.write(str(self.estimatedLipschitz[envId][repeat_list[i]][self.horizon-1]))

    return fig 

def plotLipschitz_restrict_bar(self, filepath, envId=0):
    fig = plt.figure()
    count = 1
    last_pull = np.transpose(self.lastPulls[envId][1]) # repetition, arm
    for repeatId in range(self.repetitions):
        if self.estimatedLipschitz[envId][repeatId][self.horizon-1] < ref_value:
            count +=1
            X = np.array([i + 0.8*count for i in range(self.envs[envId].nbArms)])
            Y = last_pull[repeatId]
            plt.bar(X, Y, label=self.policies[1].__cachedstr__+", constant: "+str(self.estimatedLipschitz[envId][repeatId][self.horizon-1])+", repeatId: "+str(repeatId))
    legend()
    X = np.arange(self.envs[envId].nbArms)
    plt.xticks([i*count + (1/count) for i in range(self.envs[envId].nbArms)], X)
    plt.xlabel('Ordering of arms')
    plt.ylabel('Number of arms pulls')
    plt.title("Lipschitz Constant Histogram")
    plt.savefig(filepath+'/Lipschitz_Constant_restrict_bar', dpi=300)
    return fig 

def plotLipschitzConstant(self, filepath, envId=0):
    # plot estimated Lipschitz constant => only Lipschitz_unknown case
    X = self._times - 1
    fig = plt.figure()
    
    trueLCs = np.full(self.horizon, trueLC)
    plt.plot(X, trueLCs, label='true Lipschitz Constant')

    meanLC = self.estimatedLipschitz[envId].mean(axis=0)
    std = np.std(self.estimatedLipschitz[envId], axis=0)/np.sqrt(self.repetitions)
    ci = 1.96*std
    plt.plot(X, meanLC, label='estimated Lipschitz Constant', color='green')
    plt.fill_between(X, meanLC-ci, meanLC+ci, alpha=0.3, color='green')
    
    plt.xlabel('time')
    plt.ylabel('Lipschitz Constant')
    plt.legend()
    plt.savefig(filepath+'/LipschitzConstant', dpi=300)
    return fig

def plotConditionalExpectation(self, filepath, envId=0):
    Q = defaultdict(list)
    fig = plt.figure()
    # just for estimated Lipschitz
    cum_pullarm = np.zeros((self.repetitions, self.envs[envId].nbArms))
    for repeatId in range(self.repetitions):
        for t in range(self.horizon):
            cum_pullarm[repeatId] += self.allPulls_rep[envId][repeatId, 1, :, t]
        estimated_value = self.estimatedLipschitz[envId][repeatId][self.horizon-1]
        if estimated_value < ref_value:
            ratio_suboptimal = cum_pullarm[repeatId][0] / self.horizon #sub_optimalarm
            Q[estimated_value].append(ratio_suboptimal)
    """
    # for small Lipschitz Constant
    X2 = []
    Z = []
    for i, key in enumerate(Q.keys()):
        if key < 2:
            X2.append(key)
            meanZ = np.array(list(Q.values())[i]).mean(axis=0)
            Z.append(meanZ)
    plt.bar(X2, Z, width=0.002)
    legend()
    plt.xlabel('Estimated Lipschitz Constant')
    plt.ylabel('Ratio of suboptimal arms pulls')
    plt.title('The ratio of suboptimal arm pulls for each Estimated Lipschitz Constant(<2):\n Estimated Lipschitz Constant is less than {}, Total {} repetitions, {} times'.format(str(ref_value), self.repetitions, self.horizon))
    plt.savefig(filepath+'/ConditionalExpectation_forsmallvalue', dpi=300)  
    """
    with open(filepath+ "/Conditional_Expectation.txt", "w") as f:
        f.write(str(Q))

    X = list(Q.keys())
    Y = []
    for i in range(len(Q.keys())):
        meanY = np.array(list(Q.values())[i]).mean(axis=0)
        Y.append(meanY)
    plt.bar(X, Y)
    legend()
    plt.xlabel('Estimated Lipschitz Constant')
    plt.ylabel('Ratio of suboptimal arms pulls')
    plt.title('The ratio of suboptimal arm pulls for each Estimated Lipschitz Constant:\n Estimated Lipschitz Constant is less than {}, Total {} repetitions, {} times'.format(str(ref_value), self.repetitions, self.horizon))
    plt.savefig(filepath+'/ConditionalExpectation', dpi=300)   
    
    return fig

def plotConditionalExpectation_bisection(self, filepath, envId=0):
    Q = defaultdict(list)
    fig = plt.figure()
    standard_value = 20
    
    colors =['tomato', 'deepskyblue']
    cum_pullarm = np.zeros((self.repetitions, self.envs[envId].nbArms))
    for repeatId in range(self.repetitions):
        for t in range(self.horizon):
            cum_pullarm[repeatId] += self.allPulls_rep[envId][repeatId, 1, :, t]
        estimated_value = self.estimatedLipschitz[envId][repeatId][self.horizon-1]
        if estimated_value < standard_value:
            Q[0].append(cum_pullarm[repeatId])
        else:
            Q[1].append(cum_pullarm[repeatId])
    
    X = np.arange(self.envs[envId].nbArms)
    Y = []
    for i in range(len(Q.keys())):
        meanY = np.array(list(Q.values())[i]).mean(axis=0)
        Y.append(meanY)

    plt.plot(X, Y[0], label='Lipschitz Constant<20', color=colors[0])
    plt.plot(X, Y[1], label='Lipschitz Constant>20', color=colors[1])
    legend()
    plt.xlabel('Estimated Lipschitz Constant_bisection')
    plt.ylabel('Ratio of suboptimal arms pulls')
    plt.title('Mean number of times each arm pulls in ${}$ for estimated Lipschitz Constant:\n Total {} repetitions'.format(self.horizon, self.repetitions))
    plt.savefig(filepath+'/ConditionalExpectation_bisection', dpi=300)   

    return fig

def plotConditionalExpectation_bisection20(self, filepath, envId=0):
    # top and bottom 30%

    EL_repetition = defaultdict()
    percent = 0.2

    # last pull
    cum_pullarm = np.zeros((self.repetitions, self.envs[envId].nbArms))
    for repeatId in range(self.repetitions):
        # last pull
        for t in range(self.horizon):
            cum_pullarm[repeatId] += self.allPulls_rep[envId][repeatId, 1, :, t]
        # save key: repeatId, value: Estimated Lipschitz Constant
        EL_repetition[repeatId] = self.estimatedLipschitz[envId][repeatId][self.horizon-1]
    align_ELrepetition = sorted(EL_repetition.items(), key=lambda x: x[1]) # descending order of value

    # EL_repetition = np.sort(self.estimatedLipschitz[envId][:, self.horizon-1]) # Lipschitz constant for every repetion
    
    Q = defaultdict(list)
    fig = plt.figure()
    colors =['tomato', 'deepskyblue']
    for i in range(int(self.repetitions*percent)):
        Q[0].append(cum_pullarm[align_ELrepetition[i][0]]) # bottom30
        Q[1].append(cum_pullarm[align_ELrepetition[self.repetitions-i-1][0]]) #top30
    X = np.arange(self.envs[envId].nbArms)
    Y = []
    for i in range(len(Q.keys())):
        meanY = np.array(list(Q.values())[i]).mean(axis=0)
        Y.append(meanY)
    
    plt.plot(X, Y[0], label='Lipschitz Constant bottom 20%', color=colors[0])
    plt.plot(X, Y[1], label='Lipschitz Constant top 20%', color=colors[1])
    legend()
    plt.xlabel('Estimated Lipschitz Constant_bisection')
    plt.ylabel('Ratio of suboptimal arms pulls')
    plt.title('Mean number of times each arm pulls in ${}$ for estimated Lipschitz Constant:\n Total {} repetitions'.format(self.horizon, self.repetitions))
    plt.savefig(filepath+'/ConditionalExpectation_bisection20', dpi=300)   

    return fig

def plotConditionalExpectation_bisection30(self, filepath, envId=0):
    # top and bottom 30%

    EL_repetition = defaultdict()
    percent = 0.3

    # last pull
    cum_pullarm = np.zeros((self.repetitions, self.envs[envId].nbArms))
    for repeatId in range(self.repetitions):
        # last pull
        for t in range(self.horizon):
            cum_pullarm[repeatId] += self.allPulls_rep[envId][repeatId, 1, :, t]
        # save key: repeatId, value: Estimated Lipschitz Constant
        EL_repetition[repeatId] = self.estimatedLipschitz[envId][repeatId][self.horizon-1]
    align_ELrepetition = sorted(EL_repetition.items(), key=lambda x: x[1]) # descending order of value

    # EL_repetition = np.sort(self.estimatedLipschitz[envId][:, self.horizon-1]) # Lipschitz constant for every repetion
    
    Q = defaultdict(list)
    fig = plt.figure()
    colors =['tomato', 'deepskyblue']
    for i in range(int(self.repetitions*percent)):
        Q[0].append(cum_pullarm[align_ELrepetition[i][0]]) # bottom30
        Q[1].append(cum_pullarm[align_ELrepetition[self.repetitions-i-1][0]]) #top30
    X = np.arange(self.envs[envId].nbArms)
    Y = []
    for i in range(len(Q.keys())):
        meanY = np.array(list(Q.values())[i]).mean(axis=0)
        Y.append(meanY)


    plt.plot(X, Y[0], label='Lipschitz Constant bottom 30%', color=colors[0])
    plt.plot(X, Y[1], label='Lipschitz Constant top 30%', color=colors[1])
    legend()
    plt.xlabel('Estimated Lipschitz Constant_bisection')
    plt.ylabel('Ratio of suboptimal arms pulls')
    plt.title('Mean number of times each arm pulls in ${}$ for estimated Lipschitz Constant:\n Total {} repetitions'.format(self.horizon, self.repetitions))
    plt.savefig(filepath+'/ConditionalExpectation_bisection30', dpi=300)   

    return fig   


def plotConditionalExpectation_split(self, filepath, envId=0):
    Q = defaultdict(list)

    fig = plt.figure()
    split_value = 20

    cum_pullarm = np.zeros((self.repetitions, self.envs[envId].nbArms))
    for repeatId in range(self.repetitions):
        for t in range(self.horizon):
            cum_pullarm[repeatId] += self.allPulls_rep[envId][repeatId, 1, :, t]
        estimated_value = self.estimatedLipschitz[envId][repeatId][self.horizon-1]
        ratio_suboptimal = cum_pullarm[repeatId][0] / self.horizon #sub_optimalarm
        Q[estimated_value // split_value].append(ratio_suboptimal)

    X = list(Q.keys())
    Y = []
    for i in range(int(max(Q.keys()))):
        if i not in X:
            Q[i] = None

    for i in range(len(Q.keys())):
        if list(Q.values())[i] == None:
            Y.append(0)
        else:
            meanY = np.array(list(Q.values())[i]).mean(axis=0)
            Y.append(meanY) 

    X = list(Q.keys())       
    plt.bar(X, Y)
    legend()
    plt.xlabel('Estimated Lipschitz Constant_split')
    plt.ylabel('Ratio of suboptimal arms pulls')
    plt.title('The ratio of suboptimal arm pulls for each Estimated Lipschitz Constant:\n Total {} repetitions, {} times'.format(self.repetitions, self.horizon))
    plt.savefig(filepath+'/ConditionalExpectation_split', dpi=300)   
    
    return fig    
def estimatedLipschitzdata(self, filepath, envId=0):
    # estimated Lipschitz Constant
    with open(filepath+ "/estimated_Lipschitz_Constant.txt", "w") as f:
        for t in range(self.horizon):
            #if t % 1000 == 1:
            f.write('\nt={}, '.format(t))
            f.write(str(self.estimatedLipschitz[envId][0][t]))