import os
import numpy as np
import argparse
from tqdm import tqdm

from policies.Bernolli import Bernoulli
from environments.evaluator_transfer_learning import *
from OSSB_mono_2 import OSSB, LipschitzOSSB, LipschitzOSSB_true


def com_Lipschitz_constant(thetas, embeddings):
        L_values = []
        for i in range(thetas.size-1):
            L_values.append(abs(thetas[i+1]-thetas[i])/(np.linalg.norm(embeddings[i+1]-embeddings[i])))

        return np.amax(L_values)

def main(args):

    HORIZON=50000
    REPETITIONS=100
    N_JOBS=40

    # ''' bandit instances '''
    embeddings = [0, 0.995, 0.996, 0.997, 0.998, 0.999]
    arms = np.array([0.1, 0.0005, 0.0005, 0.2005, 0.0005, 0.0005])

    assert len(arms) == len(embeddings), "check arms and embeddings"

    true_LC = com_Lipschitz_constant(arms, embeddings)
    
    ENVIRONMENTS = [ 
            {   
                "arm_type": Bernoulli,
                "params": arms
            }
        ]

    POLICIES = [
            {
                "archtype":OSSB,
                "params":{}
            },
            {
                "archtype":LipschitzOSSB,
                "params":{}
            },
            LipschitzOSSB_true(nbArms=len(embeddings), embeddings=embeddings, gamma=0.001, L=true_LC)
            ,
            LipschitzOSSB_true(nbArms=len(embeddings), embeddings=embeddings, gamma=0.001, L=0.1)
        ]

    configuration = {
        "embeddings": embeddings,
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

    dir_name = args.save_dir
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    _ = evaluation.plotRegrets_DEL(filepath=dir_name, name=args.name)
    _ = evaluation.plotArmPulls(filepath=dir_name, envId=0)
    _ = evaluation.SaveHistory(filepath=dir_name)
    _ = evaluation.RegretsHistory()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", help="save name", type=str, default=0)
    parser.add_argument("--save_dir", default="./DEL/")
    args = parser.parse_args()

    main(args)
