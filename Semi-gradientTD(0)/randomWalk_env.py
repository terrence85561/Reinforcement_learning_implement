'''
Random Walk example with 1000 states from example 9.1 in Sutton's book
'''

from rl_glue import BaseEnvironment
import numpy as np
import sys


class RandomWalkEnvironment(BaseEnvironment):

    def __init__(self):
        """Declare environment variables."""

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        """
        # init current state
        self.cur_state = None
        # init reward
        self.reward = None

        
        
    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - integer
        """
        # the start state is set to 500
        self.cur_state = 499
        
        return self.cur_state

    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - integer - terminal - boolean
        Take the action, to determine the reward and is/isn't terminal
        If state < 1, terminal with reward = -1
        If state > 1000, terminal with reward = +1
        O/W not terminal and reward = 0
        """

        self.cur_state = self.cur_state + action
        if self.cur_state < 0:
            reward = -1.0
            terminal = True
            return reward,self.cur_state,terminal
        elif self.cur_state > 999:
            reward = 1.0
            terminal = True
            return reward,self.cur_state,terminal
        reward = 0.0
        terminal = False
        return reward,self.cur_state,terminal


    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass

