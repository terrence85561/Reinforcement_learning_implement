"""
Windy Gridworld problem environment using RLGlue
"""

from rl_glue import BaseEnvironment
import numpy as np

class WindEnvironment(BaseEnvironment):
    """
    windy Gridworld environment -- Excerise 6.9 from
    RL book (2nd edition)

    Note: inherit from BaseEnvironment to be sure that your Agent class implements
    the entire BaseEnvironment interface
    """

    def __init__(self):
        """Declare environment variables."""

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
        # init the wind power
        self.wind = [0,0,0,1,1,1,2,2,1,0]


        # init state variable
        self.state = None

        # init terminal state
        self.terminal = (3,7)

        # init row & col
        self.row = None
        self.col = None

        # init gridworld size
        self.maxRow = 6
        self.maxCol = 9

    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - tuple
        """
        # initial state
        self.state = (3,0)
        return self.state

    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - tuple - terminal - boolean
        """

        # get the row & col index of the last state
        self.row = self.state[0]
        self.col = self.state[1]

        # compute the next state
        self.row =self.row + action[0] - self.wind[self.col]
        self.col = self.col + action[1]

        # if hits boarder
        if self.row < 0:
            self.row = 0

        if self.row > self.maxRow:
            self.row = self.maxRow

        if self.col < 0:
            self.col = 0

        if self.col > self.maxCol:
            self.col = self.maxCol

        # update next state
        self.state = (self.row,self.col)

        if self.state == self.terminal:
            reward = 0
            self.state = None
            terminal = True
        else:
            reward = -1
            terminal = False

        return reward,self.state,terminal

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
