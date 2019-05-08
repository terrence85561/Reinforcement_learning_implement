"""
Windy Gridworld problem environment using RLGlue
"""

from rl_glue import BaseEnvironment
import numpy as np
import copy


class DynaQEnvironment(BaseEnvironment):
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
        
        # init the position of blocks
        self.blocks = {(0,7):"_",
                       (1,7):"_",
                       (2,7):"_",
                       (1,2):"_",
                       (2,2):"_",
                       (3,2):"_",
                       (4,5):"_"}

        # init state variable
        self.state = None

        # init terminal state
        self.terminal = (0, 8)

        # init row & col
        self.row = None
        self.col = None

        # init gridworld size
        self.maxRow = 5
        self.maxCol = 8

    def env_start(self):
        #self.repState = None
        #self.repCount = 0
        #self.state_track = []
        """
        Arguments: Nothing
        Returns: state - tuple
        """
        # initial state
        self.state = (2, 0)
        #self.state_track.append(self.state)
        return self.state

    def env_step(self, action):
        #print(action)
        """
        Arguments: action - integer
        Returns: reward - float, state - tuple - terminal - boolean
        """
        #print('action is ',action)
        #print('step')

        #self.action_track[self.state] = action

        # get the row & col index of the last state
        self.row = copy.deepcopy(self.state[0])
        self.col = copy.deepcopy(self.state[1])

        # compute the next state
        self.row = self.row + action[0] 
        self.col = self.col + action[1]

        # if self.row >= 0 and self.col >= 0 and self.row <= self.maxRow and self.col <= self.maxCol:
        #     if (self.row,self.col) not in self.blocks.keys():
        #         self.state = (self.row,self.col)
                
        # if self.state in self.state_track:
        #     print('action', action, 'repeat state ', self.state)
        #     self.repCount += 1
        #     self.repState = self.state
        #     print('times of repetition', self.repCount)
        # else:
        #     self.state_track.append(self.state)
        # if hits boarder
        if self.row < 0:
            self.row = 0

        if self.row > self.maxRow:
            self.row = self.maxRow

        if self.col < 0:
            self.col = 0

        if self.col > self.maxCol:
            self.col = self.maxCol
        
        # if touch blocks, stay 
        if (self.row,self.col) in self.blocks.keys():
            #print('stuck')
            #print('row,col',self.row,self.col)
            #print(self.state[0],self.state[1])
            self.row = self.state[0]
            self.col = self.state[1]
            self.state = (self.row,self.col)
            reward = -1
            terminal = False
            return reward,self.state,terminal
        # update next state
        self.state = (self.row, self.col)

        if self.state == self.terminal:
            reward = 1
            self.state = None
            terminal = True
        else:
            reward = 0
            terminal = False

        return reward, self.state, terminal

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        # if in_message == "optimal actions":
        #     print(self.action_track)
        pass
