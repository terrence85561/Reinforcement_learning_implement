"""
Purpose: To recreate the curves in Example 8.1 in textbook.
Implementation of the interaction between dyna maze's
tabular dyna-Q agent and its environment using RLGlue
"""

from rl_glue import RLGlue
from dynaQ_env import DynaQEnvironment
from dynaQ_agent import DynaQAgent
import numpy as np


if __name__ == "__main__":
    num_episodes = 50
    num_runs = 10
    count = 0
   
    output = []
    Q = []


    n_ite = [0,5,50] # planning iteration times

    for ite in n_ite:
        result = np.zeros(num_episodes)
        print("training process with {} planning step".format(ite))
        # Create and pass agent and environment objects to RLGlue
        environment = DynaQEnvironment()
        agent = DynaQAgent(ite)
        rlglue = RLGlue(environment, agent)
        del agent, environment  # don't use these anymore


        for run in range(num_runs):
            print("run number: {}\n".format(run))
            # set seed for reproducibility
            np.random.seed(run)

            # initialize RL-Glue
            rlglue.rl_init()

            # loop over episodes
            for episode in range(num_episodes):
                
                rlglue.rl_episode()
               
                result[episode] += rlglue.num_ep_steps()
                data = rlglue.rl_agent_message("Q for all states in the episode")
                
                Q.append(data)
                
                
        result = result/num_runs
        output.append(result)
        
    
    np.save("output",output)
    




    
