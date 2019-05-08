"""
Prediction agents based on Semi-gradient TD(0)
Using Tabular feature coding
"""

from rl_glue import BaseAgent
import numpy as np


class TabularAgent(BaseAgent):

    def __init__(self):
        """Declare agent variables."""

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        
        """
        self.total_state = 1000
        self.alpha = 0.5
        self.gamma = 1
        # init weight
        self.weight = np.zeros(self.total_state )
        self.action = None
        # init state feature
        self.x_s = np.zeros((self.total_state,self.total_state))
        # one hot the state feature
        self.x_s = self.onehot(self.x_s)
        # init estimate value function
        self.v_hat = None
        # init a variable to keep the last state
        self.last_state = None
    
    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        """
        # store the state passed in
        self.last_state = state
        direction = np.random.choice(['left','right'],p = [0.5,0.5])
        if direction == 'left':
            # choose go left 1 state to 100 state
            self.action = np.random.randint(-100,0)
        else:
            # choose go left 1 state to 100 state
            self.action = np.random.randint(1,101)
        return self.action
    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - integer
        Returns: action - integer
        """
        direction = np.random.choice(['left', 'right'], p=[0.5, 0.5])
        if direction == 'left':
            # choose go left 1 state to 100 state
            self.action = np.random.randint(-100,0)
        else:
            # choose go left 1 state to 100 state
            self.action = np.random.randint(1,101)
        
        # compute the td error
        Td_error = reward + self.gamma * np.dot(self.x_s[state],self.weight) - \
            np.dot(self.x_s[self.last_state],self.weight)
        
        # gradient of estimate value function becomes to feature of self.last_state
        # becuase this is linear function approximation
        self.weight = np.add(self.weight,self.alpha * Td_error * self.x_s[self.last_state])

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
        Td_error = reward + self.gamma * 0 - np.dot(self.x_s[self.last_state],self.weight)
        
        # np.add?
        self.weight = self.weight + self.alpha * Td_error * self.x_s[self.last_state]


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
            self.v_hat = np.dot(self.weight, self.x_s)
            return self.v_hat
        else:
            return "I dont know how to respond to this message!!"

    def onehot(self,matrix):
        for i in range(self.total_state ):
            matrix[i][i] = 1
        return matrix
