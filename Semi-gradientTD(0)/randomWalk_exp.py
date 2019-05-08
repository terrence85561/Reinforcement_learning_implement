import time
import numpy as np
from tabular_agent import TabularAgent
from randomWalk_env import RandomWalkEnvironment
from tile_agent import TileCodingAgent
from rl_glue import RLGlue

'''
experiment of the random walk example of Example 9.1 in Sutton's book
'''

if __name__ == "__main__":
    num_episodes = 2000
    num_runs = 30

    # load the true state_value function which has already computed by DP
    data = np.load("TrueValueFunction.npy")
    Vs = data[1:1001]

    # an array to store the output of two agents combined
    output = []

    agent_list = ['tabular', 'tile_coding']

    for item in agent_list:
        # a numpy array to hold the outputs
        result = np.zeros(int(num_episodes/10))
        #vhat_arr = []
        # Create and pass agent and environment objects to RLGlue
        if item == 'tabular':
            agent = TabularAgent()
        elif item == 'tile_coding':
            agent = TileCodingAgent()
        #agent = TabularAgent()
        #agent = TileCodingAgent()
        environment = RandomWalkEnvironment()
        rlglue = RLGlue(environment, agent)
        del agent, environment  # don't use these anymore


        
        start_time = time.time()
        print("\nBegin to execute {} agent.......\n".format(item))
        for run in range(num_runs):

            print("run number: {}\n".format(run))

            # set seed for reproducibility
            np.random.seed(run)

            # initialize RL-Glue
            rlglue.rl_init()

            # loop over episodes
            for episode in range(num_episodes):
                #print("episode{}".format(episode))
                # run episode with the allocated steps budget
                rlglue.rl_episode()
                if episode % 10 == 0:
                    V_hat = rlglue.rl_agent_message("Estimate value function")
                    #print(V_hat)
                    #vhat_arr.append(V_hat)
                    # reference: https://stackoverflow.com/questions/21926020/how-to-calculate-rmse-using-ipython-numpy
                    RMSE = np.sqrt(np.mean((Vs - V_hat)**2))
                    #print(RMSE)
                    result[int(episode/10)] += RMSE
        result = result/num_runs
        output.append(result)
        print('total time of executing 30 rums with {} agent is {:3}s'.format(item,time.time()-start_time))
        print(result)
    #np.savez('randomwalk.npz',tabular = output[0],tile_coding = output[1] )
    np.save('randomwalk', output)
