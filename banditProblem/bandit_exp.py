"""Example experiment for CMPUT 366 Fall 2019

This experiment uses the rl_episode() function.

Runs a random agent in a 1D environment. Runs 10 (num_runs) iterations of
100 episodes, and reports the total reward. Each episode is capped at 100 steps.
(max_steps)
"""
import numpy as np
from ucb_agent import Ucb_BanditAgent
from bandit_agent import BanditAgent
from bandit_env import BanditEnv
from rl_glue import RLGlue


def BanditExp(rlg, num_runs, max_steps):

    # store how many times optimal Action is picked in each time step
    optimalAction = np.zeros(max_steps)
    # totalReward = np.zeros((max_steps),dtype = float)
    #rewards = np.zeros(num_runs)
    for run in range(num_runs):
        #run 2000 times
        # set seed for reproducibility
        np.random.seed(run)

        # initialize RL-Glue
        rlg.rl_init()

        # start RL-Glue
        rlg.rl_start()
        for step in range(max_steps):
            stepsize = rlg.num_ep_steps()
            rlg.rl_agent_message(stepsize)

            # if the action in this step is optimal action, optimalAction(step) += 1
            #print(rlg.rl_step()[1])
            reward, state, action, is_terminal = rlg.rl_step()
            if action == rlg.rl_env_message("optimal action"):
                optimalAction[step] += 1

    return optimalAction


def main():
    choice = input("Enter 1 to select question3 agent, enter 2 to select bonus agent: ")
    if choice == '1':

        Q1 = float(input("Enter value of Q1: "))
        alpha = float(input("Enter value of alpha: "))
        epsilon = float(input("Enter value of epsilon: "))
        agent = BanditAgent(Q1,alpha,epsilon)
    if choice == '2':
        Q1 = float(input("Enter value of Q1: "))
        c =  float(input("Enter value of c: "))
        alpha = float(input("Enter value of alpha: "))
        agent =  Ucb_BanditAgent(Q1,c,alpha)
    name = input("Input output file name: ")

    max_steps = 1000  # max number of steps in an episode
    num_runs = 2000  # number of repetitions of the experiment

    # Create and pass agent and environment objects to RLGlue

    environment = BanditEnv()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore

    # run the experiment
    optimalAction = BanditExp(rlglue, num_runs, max_steps)
    result = optimalAction / num_runs
    print(result)
    with open(name+'.csv','w') as out_file:
        for i in range(max_steps):
            out_file.write("%f\n" %result[i])


if __name__ == '__main__':
    main()
