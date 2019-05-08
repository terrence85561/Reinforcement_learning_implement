import numpy as np
from rl_glue import BaseAgent


class BanditAgent(BaseAgent):
    """
    simple random agent, which moves left or right randomly in a 2D world

    Note: inheret from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self,Q1,alpha,epsilon):
        """Declare agent variables."""

        # Your agent may need to remember what the action taken was.
        # In this case the variable is not used.
        self.prevAction = None
        self.Q1 = Q1
        self.alpha = alpha
        self.epsilon = epsilon



        # Your agent may have a policy for choosing actions.
        # This agent will exploit with 1-self.epsilon probability
        # will explore with epsilon probability

    def agent_init(self):
        """Initialize agent variables."""
        #self.epsilon = 0.1
        #self.Q1 = 0
        #self.alpha = 0.1
        self.arm = 10
        self.Qn = np.zeros(self.arm) + self.Q1



    def _choose_action(self):
        """
        Convenience function.

        You are free to define whatever internal convenience functions
        you want, you just need to make sure that the RLGlue interface
        functions are also defined as well.
        """

        choice = np.random.choice([0,1],p = [1-self.epsilon,self.epsilon])
        if choice == 0:
            #exploiting
            action = np.argmax(self.Qn)
        if choice == 1:
            action = np.random.randint(self.arm)
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
        pass
