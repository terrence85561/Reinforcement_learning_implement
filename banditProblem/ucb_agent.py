import numpy as np
from rl_glue import BaseAgent
from math import sqrt,log


class Ucb_BanditAgent(BaseAgent):
    """
    simple random agent, which moves left or right randomly in a 2D world

    Note: inheret from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self,Q1,c,alpha):
        """Declare agent variables."""


        self.prevAction = None
        self.Q1 = Q1
        self.c = c
        self.alpha = alpha


    def agent_init(self):
        """Initialize agent variables."""
        #self.epsilon = 0
        #self.Q1 = 1
        #self.alpha = 0.1
        self.arm = 10
        self.Qn = np.zeros(self.arm) + self.Q1
        self.Na = np.zeros(self.arm) # to store number of times that a certain action has been selected

        self.ucb = None




    def _choose_action(self):
        """
        Convenience function.

        You are free to define whatever internal convenience functions
        you want, you just need to make sure that the RLGlue interface
        functions are also defined as well.
        """

        # when the action is not taken before, choose that action
        for i in range(self.arm):
            if self.Na[i] == 0:
                action = i
                return action


        ucb = self.ucb_builder(self.Qn,self.step,self.Na)

        action = np.argmax(ucb)




        return action

    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (state observation): The agent's current state

        Returns:
            The first action the agent takes.
        """

        # This agent doesn't care what state it's in, it always chooses
        # to move left or right randomly according to self.probLeft
        # self.prevAction = self._choose_action()

        # start from exploring
        self.prevAction = np.random.randint(self.arm)
        return self.prevAction

    def agent_step(self, reward, state):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (state observation): The agent's current state
        Returns:
            The action the agent is taking.
        """
        self.Qn[self.prevAction] = self.Qn[self.prevAction] + self.alpha * (reward - self.Qn[self.prevAction])


        self.prevAction = self._choose_action()
        self.Na[self.prevAction] += 1




        return self.prevAction

    def agent_end(self, reward):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """

        # random agent doesn't care about reward
        pass

    def agent_message(self, message):
        self.step = message

    def ucb_builder(self,Qn,step,Na):
        self.ucb = np.zeros(self.arm)
        #build ucb
        # for each action, update the matrix respect to current step and
        # the number of time that the action was taken
        for i in range(self.arm):
            self.ucb[i] = Qn[i] + self.c * sqrt(log(step)/Na[i])
        return self.ucb
