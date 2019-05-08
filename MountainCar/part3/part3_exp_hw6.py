#!/usr/bin/env python

import numpy as np
from part3_agent_hw6 import Agent
import time
from rl_glue import RLGlue
from part3_env_hw6 import Environment



def question_1():
    # Specify hyper-parameters

    agent = Agent()
    environment = Environment()
    rlglue = RLGlue(environment, agent)

    num_episodes = 1000
    num_runs = 1
    max_eps_steps = 100000
    
    num_action = 3

    
    
    for r in range(num_runs):
        print("run number : ", r)
        #np.random.seed(r)
        rlglue.rl_init()
        for _ in range(num_episodes):
            rlglue.rl_episode(max_eps_steps)
        weight = rlglue.rl_agent_message('get weight')
     
    # algorithm from assignment 
    #fout = open('value','w')
    steps = 50
    neg_q_hat = np.zeros((steps,steps))
    for i in range(steps):
        for j in range(steps):
            values = []
            position = -1.2+(i*1.7/steps)
            velocity = -0.07+(j*0.14/steps)
            for a in range(num_action):
                tile_idx = agent.plot_get_feature(position,velocity,a)
                q_hat = np.sum(weight[tile_idx])
                values.append(q_hat)
            height = np.max(values)
            neg_q_hat[j][i] = -height
            #fout.write(repr(-height)+' ')
        #fout.write('\n')
    #fout.close()
    np.save('neg_q_hat',neg_q_hat)

                

if __name__ == "__main__":
    question_1()
    print("Done")
