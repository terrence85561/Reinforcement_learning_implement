"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018,
  University of Alberta.
  Gambler's problem environment using RLGlue.
"""
from rl_glue import BaseEnvironment
import numpy as np
import sys


class GamblerEnvironment(BaseEnvironment):
    """
    Slightly modified Gambler environment -- Example 4.3 from
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
        # probability of getting head
        self.ph = 0.5
        self.cur_state = None
    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        Hint: Sample the starting state necessary for exploring starts and return.
        """
        # exploring start, randomly give a state from 1 to 99 to begin the episode
        self.cur_state = np.zeros(1,dtype = np.int32) + np.random.randint(1,100)


        return self.cur_state

    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - terminal - boolean
        Hint: Take a step in the environment based on dynamics; also checking for action validity in
        state may help handle any rogue agents.
        """

        # check for action validity
        if action > min(self.cur_state[0],100-self.cur_state[0]) or action <=0 :
            print('current state is', self.cur_state)
            print('the selected action is', action)
            print("An invalid action occurs.........")
            sys.exit(0)


        # based on ph, calculate the next state, reward, or bool(terminal)
        # prob = 0 means tail, prob = 1 means head
        prob = np.random.choice([0,1],p = [1 - self.ph, self.ph])
        #print("now state is ",self.cur_state,'action is',action)
        if prob == 0:
            self.cur_state[0] = self.cur_state[0] -  action

            if self.cur_state[0] <= 0:
                #print("lost !!!!")
                #terminate, lose all captial
                reward = 0.0
                self.cur_state = None
                terminal = True
                return reward,self.cur_state,terminal
        elif prob == 1:

            self.cur_state[0] = self.cur_state[0] + action

            if self.cur_state[0] >= 100:
                #print("win!!!")
                # terminate, win!!!!
                reward = 1.0
                self.cur_state = None
                terminal = True
                return reward, self.cur_state,terminal
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
