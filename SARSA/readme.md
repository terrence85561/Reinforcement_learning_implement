Use python3 to run wind_exp.py
Then use python3 to run plot.py to get the plot

For 8 action result, comment out the 43th line in sarsa_agent.py
For 9 action result, uncomment the 43th line in sarsa_agent.py 

Q1(c)
  Set alpha as 0.5 and epsilon as 0.1.Comparing with the origin 4 movement situation on textbook,
  the agent can complete 349 Episodes in 8000 steps. Whereas 4 movement agent can only complete
  around 170 Episodes in 8000 steps.
  I have tried various alpha and epsilon. In 8 movement situation, with epsilon = 0.2, alpha = 0.5,
  the agent can complete 225 Episodes in 8000 steps, less than 349 Episodes.
  I also found that, with epsilon = 0.1, alpha = 0.3, the agent can only complete 203 Episodes in
  8000 steps. The largest amount of Episodes that the agent can complete in 8000 steps is 392, with
  epsilon = 0.1, alpha = 0.7. But, if alpha larger than 0.7, the number of completed Episodes decreases.
  Therefore, the best epsilon selection is 0.1 and best alpha selection is 0.7. Besides, alpha cannot
  be too low or too high.
  With respect to 9 movement agent, it performs best with epsilon = 0.1, alpha = 0.6 (complete 307 Episodes).
  Whatever the parameter changes, 9 movement agent cannot perform better than 8 movement agent. But it is
  better than 4 movement agent with same parameter. The reason for worse performance of 9 movement agent may
  be there is one more action to be chosen while doing exploring. In addition, the 9th action 'stay' is not
  a good action in our situation which the terminal state is (3,7) since this action cannot cross the windy
  area and reach the terminal state.
