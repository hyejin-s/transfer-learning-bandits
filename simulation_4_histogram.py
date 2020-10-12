from Bernolli import Bernoulli
from OSSB import OSSB, OSSB_DEL, LipschitzOSSB, LipschitzOSSB_true, LipschitzOSSB_DEL, LipschitzOSSB_DEL_true, arms
from evaluator import *
from tqdm import tqdm
import os


import numpy as np

HORIZON=1000
REPETITIONS=1 #plotArmPulls10 more than 10
N_JOBS=1


ENVIRONMENTS = [ 
        {   
            "arm_type": Bernoulli,
            "params": arms
        }
    ]

# should fix estimated Lipshitz order 1 (for store)
POLICIES = [
        {
            "archtype":LipschitzOSSB_DEL_true,
            "params":{}
        }
    ]
"""
,
        {
            "archtype":LipschitzOSSB_DEL,
            "params":{}
        },
        {
            "archtype":LipschitzOSSB,
            "params":{}
        }
"""

configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)num
    "repetitions": REPETITIONS,
    # --- Parameters for the use of joblib.Parallel
    "n_jobs": N_JOBS,    # = nb of CPU cores
    "verbosity": 6,      # Max joblib verbosity
    # --- Arms
    "environment": ENVIRONMENTS,
    # --- Algorithms
    "policies": POLICIES,
}

evaluation = Evaluator(configuration)

for envId, env in tqdm(enumerate(evaluation.envs), desc="Problems"):
    # Evaluate just that env
    evaluation.startOneEnv(envId, env)

dir_name = "simulation_new/test"
if not os.path.exists(dir_name):
    os.makedirs(dir_name)


_ = evaluation.plotRegrets(filepath="simulation_new/test")
_ = evaluation.plotRegrets(filepath="simulation_new/test", semilogy=True)
_ = evaluation.plotArmPulls(filepath="simulation_new/test")
_ = evaluation.plotLipschitzConstant(filepath="simulation_new/test")

_ = evaluation.savedata(filepath="simulation_new/test")
"""
_ = evaluation.plotLipschitz_hist(filepath="simulation_new/test")
_ = evaluation.plotConditionalExpectation(filepath="simulation_new/test")
_ = evaluation.plotConditionalExpectation_bisection(filepath="simulation_new/test")
_ = evaluation.plotConditionalExpectation_bisection20(filepath="simulation_new/test")
_ = evaluation.plotConditionalExpectation_bisection30(filepath="simulation_new/test")
_ = evaluation.plotConditionalExpectation_split(filepath="simulation_new/test")

_ = evaluation.plotLipschitz_restrict(filepath="simulation_new/test")
"""
