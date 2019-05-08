"""
Prediction agents based on semi-gradient Sarsa(lambda)
Using tile coding
"""

from rl_glue import BaseAgent
import numpy as np
from tile3 import *


class Agent(BaseAgent):

    def __init__(self):
        """Declare agent variables."""

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        
        """
        # init the number of tilings to 8
        self.num_tiling = 8
        
        self.gamma = 1
        self.lam = 0.9
        
        self.max_size = 2048 

        # init index hash table
        self.iht = IHT(self.max_size)
        
        # define epsilon
        self.epsilon = 0
        # define alpha
        self.alpha = 0.1/self.num_tiling

        # init weight
        self.weight = np.random.uniform(-0.001,0,self.max_size)
        
        self.num_action = 3
        self.action = None
        
        # init a variable to keep the last state
        self.last_state = None
        
        # init position range
        self.pos_range = 0.5 + 1.2
        # init velocity range
        self.vel_range = 0.07 + 0.07
        # tile shape
        self.tile_shape = [8,8]
        # init scaleFactor in tile coding
        self.pos_scaleFactor = self.tile_shape[0] / self.pos_range
        self.vel_scaleFaction = self.tile_shape[1] / self.vel_range
       
        # init a track for store the states have been tile coding
        self.track = {}

    def agent_start(self, state):
        """
        Arguments: state - numpy array - [position,velocity]
        Returns: action - integer
        """
        # init eligibility trace, same shape as weight
        self.z = np.zeros(self.max_size)
        
        # get action by epsilon-greddy
        # for greedy: the action = argmax_a(dot(F(s,a),weight)
        self.action = self.choose_action(state)
        
        # store current state for function update
        self.last_state = state
        
        self.last_tile_idx = None

        return self.action

    def agent_step(self, reward, state):
        
        """
        Arguments: reward - floting point, state - integer
        Returns: action - integer
        """
        
        # set td error equals to reward
        td_error = reward 
        
        # loop for i in F(s,a)
        # replacing trace
        for i in self.get_feature(self.last_state,self.action):
            td_error = td_error - self.weight[i]
            self.z[i] = 1

        # choose next action ~ q_hat
        cur_action = self.choose_action(state)

        # loop for i in F(s',a')
        tile_idx = self.get_feature(state,cur_action)
        for i in tile_idx:
            td_error = td_error + self.gamma * self.weight[i] # weight type int float?
        
        # update weight
        self.weight = self.weight + self.alpha * td_error * self.z
        # update Et
        self.z = self.gamma * self.lam * self.z 

        self.last_state = state
        self.action = cur_action
        self.last_tile_idx = tile_idx
        
        return self.action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Do the last update of weight when episode end
        """
        
        td_error = reward
        for i in self.last_tile_idx:
            td_error = td_error - self.weight[i]
            self.z[i] = 1
        
        self.weight = self.weight + self.alpha * td_error * self.z

        

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'get weight':
            return self.weight
        else:
            return "I dont know how to respond to this message!!"


    def choose_action(self,state):
        #status = self.has_titled(state)
        #if not status:
            #self.track[state] = "_"
           
        choice = np.random.choice([0,1],p=[self.epsilon,1-self.epsilon])
        if choice == 0:
            print('exploring')
            action = np.random.randint(0,3)
            return action
        
        elif choice == 1:
            value_arr = []
            # get the action with max value
            for act in range(self.num_action):
                #q_hat = 0
                # it is a array with len of 2048
                tile_idx = tiles(self.iht,self.num_tiling,[float(state[0]*self.pos_scaleFactor),
                                                    float(state[1]*self.vel_scaleFaction)],[act])
                # for idx in tile_idx:
                #     q_hat += self.weight[idx]
                q_hat = np.sum(self.weight[tile_idx])
                value_arr.append(q_hat)
            # call tie breaker function
            action = self.my_argmax(value_arr)
            #action = np.argmax(value_arr)
            
            return action


    def my_argmax(self,arr):
        max_value = np.max(arr)
        max_arg = []
        for i in range(len(arr)):
            if arr[i] == max_value:
                max_arg.append(i)
        if len(max_arg) == 0:
            return max_arg[0]
        else:
            return np.random.choice(max_arg)
 
    def plot_get_feature(self,pos,vel,action):
        # this is a function will be called in experiment for plotting 3d graph
        # pos is scaled position,vel is scaled velocity
        tile_idx = tiles(self.iht,self.num_tiling,[float(pos*self.pos_scaleFactor),
                        float(vel*self.vel_scaleFaction)],[action])
        return tile_idx    
    
    def get_feature(self,state,action):
        status = self.has_titled(state,action)
        state = tuple(state)
        if not status:
            tile_idx = tiles(self.iht,self.num_tiling,[float(state[0]*self.pos_scaleFactor),
                            float(state[1]*self.vel_scaleFaction)],[action])
            self.track[(state,action)] = tile_idx
        else:
           tile_idx = self.track[(state,action)]
        return tile_idx
    
    def has_titled(self,state,action):
        state = tuple(state)
        if (state,action) in self.track.keys():
            return True
        else:
            return False
