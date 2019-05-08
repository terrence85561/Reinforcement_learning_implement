"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np


class MonteCarloAgent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.number_state = 99
        # set two dummy state, 0 and 100 as the lost state and win state
        # initialize the policy of 1-99 state
        self.pi = np.zeros(self.number_state+2)
        for i in range(1,self.number_state+1):
            self.pi[i] =  min(i,100-i)

        #print("the policy list contains{}".format(self.pi))

        # initialize Q(s,a)
        # also, set two dummy state and a dummy action
        self.Q = np.zeros((self.number_state+2,50+1))

        # initialize a dict to store the return of each state_action pairs
        self.Return = {}

        self.Gt = None

        self.gamma = 1 # in case of requiring in future

        self.action = None

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        # initialize a track list to store every state_action pairs in an episode
        self.Track = []

        # initialize a list to store all the rewards corresponding to every state_action pairs
        self.Reward = []

        # exploring start, giving a random action from 1 to state or 100-state
        self.action = np.random.randint(1,min(state,100-state)+1)
        #print('initial state is {}, I take a random action {}'.format(state,self.action))

        # store this action_state pair in track as tuple
        self.Track.append((state[0],self.action))
        return self.action
    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        # store the reward in the reward list
        self.Reward.append(reward)

        # generate a action based on pi
        self.action = int(self.pi[state[0]])

        # save this state_action pair to track list as tuple
        self.Track.append((state[0],self.action))

        #print('in agent_step, current state is {}, the action is {}'.format(state,self.action))
        return self.action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        # self.Track stores all state_action pairs in this episode
        # self.Reward stores all the corresponding reward
        # self.Return is a dict to store the corresponding expected return for each state_action pair
        # append the last reward into reward list
        self.Reward.append(reward)
        self.Gt = 0

        # travese the track list reversely
        for i in range(len(self.Track)-1,-1,-1):
            # calculate backward
            self.Gt = self.Reward[i] + self.gamma * self.Gt
            # store it to Return dict with last state_action pair in the Track as the key
            key_of_return = self.Track[i]
            # to judge if the state_action pair occurs before
            boolean =  self.is_validate(key_of_return,i, self.Track)
            if True:
                if key_of_return in self.Return:
                    #print('same key!!!!!!!!')
                    self.Return[key_of_return].append(self.Gt)
                    #print(self.Return[key_of_return])
                else:
                    self.Return[key_of_return] = [self.Gt]

                self.Q[key_of_return[0]][key_of_return[1]] = sum(self.Return[key_of_return])/len(self.Return[key_of_return])
                # I think we need to handle the situation that the agent lost in first epsiode
                # and get return as 0. i.e. argmax(self.Q[state]) == 0
                # or handle the situation when there are more than 1 max value
                # so I wrote a custom argmax function
                # improve policy
                self.pi[self.Track[i][0]] = self.my_argmax(self.Track[i][0],self.Q[self.Track[i][0]])

            else:
                continue
        #print(self.pi)
    def is_validate(self,state_action,idx,history):
        # arguments
        # state_action: the current state_action pair
        # idx : the index of the current state_action pair in history list
        # history: the state_action pair list
        for i in range(1,idx):
            if history[i] == state_action:
                return False
        return True

    def my_argmax(self,state,list):
        max_value = np.max(list)
        max_args = []
        #for i in range(0,len(list)):
        for i in range(0,min(state,100-state)+1):
            if list[i] == max_value:
                max_args.append(i)
        if len(max_args) > 1 :
            random_best_action = np.random.choice(max_args[1:])
            return random_best_action
        best_action = max_args[0]
        return best_action

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
            return (np.max(self.Q, axis=1)).tostring()
        else:
            return "I dont know how to respond to this message!!"
