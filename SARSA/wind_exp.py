"""
Purpose: To complete the Exercise 6.9 in textbook.
Implementation of the interaction between windy Gridworld's
SARSA agent and its environment using RLGlue
"""

from rl_glue import RLGlue
from wind_env import WindEnvironment
from sarsa_agent import SarsaAgent
import numpy as np

if __name__ == "__main__":
    max_total_step = 8000
    total_num_episode = [0]
    current_step = [0]

    # Create and pass agent and environment objects to RLGlue
    environment = WindEnvironment()
    agent = SarsaAgent()
    rlglue = RLGlue(environment,agent)
    del agent,environment # don't use these anymore

    # set seed for reproducibility
    np.random.seed(1)

    # initialize RL-Glue
    rlglue.rl_init()

    # loop for the experiment step less that 8000
    while (rlglue.num_steps() < max_total_step ):
        rlglue.rl_episode(max_total_step)
        total_num_episode.append(rlglue.num_episodes())
        current_step.append(rlglue.num_steps())
    print(total_num_episode)
    np.savez('windy.npz',timeSteps = current_step,Episodes = total_num_episode)
    # to load:
    # data = np.load('windy.npz')
    # print(data["timeSteps"]) to get the array
