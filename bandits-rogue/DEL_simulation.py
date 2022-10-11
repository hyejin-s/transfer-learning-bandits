from Bernolli import Bernoulli
from Thompson import Thompson
from UCB import UCB
from klUCB import klUCB
from OSSB import  arms, embeddings, OSSB, OSSB_DEL, LipschitzOSSB, LipschitzOSSB_true, LipschitzOSSB_DEL, LipschitzOSSB_DEL_true
from evaluator import *
# from DEL_Algorithm1 import DEL_bandit, arms, embeddings
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


import numpy as np

HORIZON=1000
REPETITIONS=10 #plotArmPulls10 more than 10
N_JOBS=40

ENVIRONMENTS = [ 
        {   
            "arm_type": Bernoulli,
            "params": arms
        }
    ]

# POLICIES.append(LipschitzOSSB_DEL_true(nbArms=6, gamma=0.001, L=trueLC))
# should fix estimated LipshitDEL/0103_test5*estimatioe)
POLICIES = [
        {
            "archtype":OSSB_DEL,
            "params":{}
        },
        {
            "archtype":LipschitzOSSB_DEL,
            "params":{}
        },
        LipschitzOSSB_DEL_true(nbArms=6, gamma=0.001, L=200)
        ,
        LipschitzOSSB_DEL_true(nbArms=6, gamma=0.001, L=0.1)
    ]


configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)num
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 100,      # Max joblib verbosity
    # --- Arms
    "environment": ENVIRONMENTS,
    # --- Algorithms
    "policies": POLICIES,
}

evaluation = Evaluator(configuration)

for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
    # Evaluate just that env
    evaluation.startOneEnv(envId, env)


save_dir = "./DEL"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

_ = evaluation.plotRegrets_DEL(filepath=save_dir)
_ = evaluation.plotRegrets(filepath=save_dir)

'''
_ = evaluation.plotRegrets_log(filepath=save_dir)
_ = evaluation.plotArmPulls(filepath=save_name)
_ = evaluation.plotLipschitzConstant(filepath=save_dir)
_ = evaluation.plotLipschitzConstant_tb(filepath=save_dir)
_ = evaluation.savedata(filepath=save_name)

_ = evaluation.plotLipschitz_hist(filepath=save_dir)
_ = evaluation.plotConditionalExpectation(filepath=save_dir)

_ = evaluation.plotConditionalExpectation_bisection(filepath=save_dir)
_ = evaluation.plotConditionalExpectation_topbottom(filepath=save_dir)
_ = evaluation.plotConditionalExpectation_bottom(filepath=save_dir)

_ = evaluation.plotConditionalExpectation_split(filepath=save_dir)
'''