"""
Prediction agents based on Semi-gradient TD(0)
Using tile coding
"""

from rl_glue import BaseAgent
import numpy as np
from tiles3 import IHT,tiles


class TileCodingAgent(BaseAgent):

    def __init__(self):
        """Declare agent variables."""

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        
        """
        # init the number of tilings to 50
        self.num_tiling = 50
        self.total_state = 1000
        self.gamma = 1

        self.tile_width = 0.2
        # tile width is 0.2
        # 5 tiles cover 1000 states
        # add 1 tile for offsetting
        # 6 * num_tilings = 300
        self.max_size = int(((1/self.tile_width) + 1 ) * self.num_tiling) 

        # init index hash table
        self.iht = IHT(self.max_size)
        

        # define alpha
        self.alpha = 0.01/self.num_tiling

        # init weight
        self.weight = np.zeros(self.max_size)
        
        self.action = None
        
        # init a variable to keep the last state
        self.last_state = None

        # init state feature
        # for every row, contains binary features obtained by tile coding
        # each row for a single state 
        self.x_s = np.zeros((self.total_state, self.max_size))

        # init estimate state value function
        self.v_hat = None
        
        # init a track for store the states have been tile coding
        self.track = {}

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        """
        # store the state passed in
        self.last_state = state
        direction = np.random.choice(['left', 'right'], p=[0.5, 0.5])
        if direction == 'left':
            # choose go left 1 state to 100 state
            self.action = np.random.randint(-100, 0)
        else:
            # choose go left 1 state to 100 state
            self.action = np.random.randint(1, 101)
        return self.action

    def agent_step(self, reward, state):
        # if state == 0:
        #     print( 'state 0 occurs')
        #     exit(1)
        """
        Arguments: reward - floting point, state - integer
        Returns: action - integer
        """
        direction = np.random.choice(['left', 'right'], p=[0.5, 0.5])
        if direction == 'left':
            # choose go left 1 state to 100 state
            self.action = np.random.randint(-100, 0)
        else:
            # choose go left 1 state to 100 state
            self.action = np.random.randint(1, 101)

        # get the state's feature
        #self.get_feature(state)
        # compute the td error
        Td_error = reward + self.gamma * np.dot(self.get_feature(state), self.weight) - \
            np.dot(self.get_feature(self.last_state), self.weight)

        # gradient of estimate value function becomes to feature of self.last_state
        # becuase of this is linear function approximation
        self.weight = np.add(self.weight, self.alpha *
                             Td_error * self.get_feature(self.last_state))
        self.last_state = state

        return self.action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Do the last update of weight when episode end
        """
        # compute the td error
        # here is the end of episode, next state does not exist
        Td_error = reward + self.gamma * 0 - \
            np.dot(self.get_feature(self.last_state), self.weight)

        # np.add?
        self.weight = self.weight + self.alpha * \
            Td_error * self.get_feature(self.last_state)

        

        #self.v_hat = self.v_hat[0]
        #print(self.v_hat)
        #print(self.v_hat.shape)

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'Estimate value function':
            # compute the estimate value function
            self.v_hat = np.dot(self.x_s, self.weight)
            return self.v_hat
        else:
            return "I dont know how to respond to this message!!"


    def get_feature(self,state):
        status = self.has_titled(state)
        if not status:
            self.track[state] = "_"
           
            # tile width is 0.2
            # 5 tiles cover 1000 states
            scaleFactor = ((1/self.tile_width)) / self.total_state
            # it is a array with len of 50
            tile_idx = tiles(self.iht,self.num_tiling,[float(state*scaleFactor)])
            
            #print(tile_idx)
            for idx in tile_idx:
                self.x_s[state][idx] = 1
            #print(self.x_s[state])
            # return self.x_s[state]
        return self.x_s[state]
    
    def has_titled(self,state):
        if state in self.track.keys():
            return True
        else:
            return False
