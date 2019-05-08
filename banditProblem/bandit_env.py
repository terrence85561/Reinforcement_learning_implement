from rl_glue import BaseEnvironment
import numpy as np

class BanditEnv(BaseEnvironment):
    """
    Example 1-Dimensional environment
    """
    qstart = None
    arm = 10
    def __init__(self):
        """Declare environment variables."""

        # number of valid states
        self.numStates = None

        # state we always start in
        self.startState = None

        # state we are in currently
        self.currentState = None

        # possible actions
        self.actions = [-1, 1]



    def env_init(self):
        #global qstart,arm
        """
        Initialize environment variables.
        """

        # self.numStates = 10
        # self.startState = 5
        # obtain q* for ten arms obey normal distribution
        self.arm = 10
        #for i in range(self.arm):
        self.qstart = np.random.normal(0.0,1.0,self.arm)



    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        self.currentState = self.startState
        return self.currentState

    def env_step(self, action):
        #global qstart
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """


        # R ~ N(q*(a),1)
        terminal = False

        reward = np.random.normal(self.qstart[action],1.0)

        return reward, self.currentState, terminal

    def env_message(self, message):
        if message == 'optimal action':

            # return the action with highest value

            return np.argmax(self.qstart)
