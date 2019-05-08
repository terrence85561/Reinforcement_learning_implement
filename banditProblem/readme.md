In my submission, there are two agent file: bandit_agent.py and ucb_agent.py
                            one environment file: bandit_env.py
                            one experiment file: bandit_exp.py
                            one graph plotting file: plot.py
                            two graph: 1."Q1=0" combines "Q1=5"
                                       2."ucb"

Put all these files in one folder, and run bandit_exp.py use python3, then follow
the prompts.Then run plot.py to plot the graphs.

Steps:
For question3 in the assignment1:
  1. run bandit_exp.py use python3, choose "1" for this question.
  2. Enter the value of Q1, alpha, epsilon, and the output file name.
     This step need to do twice in order to make two output files, one for epsilon-greedy method, one for optimal initial value method.
     E.g."Q1=0,epsilon=0.1,alpha = 0.1, output file name = Q1=0"
     and "Q1 = 5, epsilon = 0,alpha = 0.1, output file name = Q1=5" respectively.
  3. Run plot.py, select "E" as epsilon-greedy method, enter two input file names
     to plot the graph.(E.g "Q1=0","Q1=5")

For bonus question using the UCB method:
  1. run bandit_exp.py use python3, choose "2" for ucb.
  2. Enter value of Q1,alpha,and c.
  3. Run plot.py, select "U" as ucb method and enter the input file name you just made.

Report of bonus question:
  I chose Q1 = 2 and c = 0.1 for the ucb method. In this choice, I think my ucb agent performs best.
  I found that the choose of the value of Q1 and c have some relation. When initial value increase, the confident parameter should decrease to get a better learning performance.
  In my case, my ucb_agent can reach roughly up to 87% optimal action. Whereas the epsilon-greedy agent with Q1=0,epsilon = 0.1 can only reach 76%. Hence, ucb_agent is outperform.
  Compare with epsilon-greedy method or the optimal initial value method, ucb agent explores more since the arms which are chosen less, will have more chance to be chosen.
