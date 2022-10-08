from UCB1_SMPy import UCB
from policies.Bernolli import Bernoulli
# from OSSB_SMPy import OSSB_SMPy
# from OSSB import OSSB, LipschitzOSSB

from OSSB_transfer_hyp10 import OSSB_DEL, LipschitzOSSB_DEL, LipschitzOSSB_DEL_true
from evaluator import *
import matplotlib
from tqdm import tqdm


import numpy as np

HORIZON=10000
REPETITIONS=50
N_JOBS=1

#arms = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

arms10 = np.array([0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

arms50 = np.array([0.1])
for i in range(51):
    arms50 = np.append(arms50, [0.001])

ENVIRONMENTS = [ 
        {   
            "arm_type": Bernoulli,
            "params": arms10
        }
    ]

POLICIES = [
        # {
        #     "archtype":UCB,
        #     "params":{}
        # },
        {
            "archtype":OSSB_DEL,
            "params":{}
        },
        # {
        #     "archtype":LipschitzOSSB,
        #     "params":{}
        # },
    ]



configuration = {
    # --- Duration of the experiment
    "horizon": HORIZON,
    # --- Number of repetition of the experiment (to have an average)
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



_ = evaluation.plotRegrets()