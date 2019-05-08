import numpy as np
import matplotlib.pyplot as plt

def sweep_plot(yplot,sweep):
    plt.title('ph = {}'.format(ph))
    for i in range(4):
        if i < 3:
            plt.plot(yplot[i][:], label = 'sweep {}'.format(i+1))
        else:
            plt.plot(yplot[i][:],label = 'sweep {}'.format(sweep))
        plt.xlim([1,99])
        plt.xticks([1,25,50,75,99])
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
        plt.legend()
    plt.show()

def oplicy_plot(bestAction):
    #plot graph
    plt.title('ph = {}'.format(ph))
    x = np.arange(1,100)
    y = bestAction[1:100]
    plt.plot(x,y)
    plt.xticks([1,25,50,75,99])
    plt.xlabel('Captial')
    plt.ylabel('Final Policy(stake)')
    plt.show()

def maxValue(state,Vs):
    action = np.arange(1,min(state,100-state)+1)
    curVs = np.zeros(action.shape) # store the value for each action

    for i in action:
        if state + i >= 100:
            # reach the goal
            curVs[i-1] = ph * (1 + Vs[state+i]) +(1-ph) * (0 + Vs[state-i])
        else:
            # normal situation
            curVs[i-1] = ph * (0 + Vs[state+i]) +(1-ph) * (0 + Vs[state-i])
    return max(curVs)





#global var
ph = 0.25           # prob for head
theta = 1e-18  # a small threshold to terminate the loop
sweep = 0        # record number of sweep
delta = 1        # a number to compare with theta
y_plot = np.zeros((4,99))

state = np.arange(0,102) # holds all state, including two dummy state:
                         # state[0] and state[101] as terminal state

# init V(s)
Vs = np.zeros(state.shape)
Vs[-1] = 1 # set the last state as the terminal state while gambler reach $100

bestAction = np.zeros(state.shape) # to store the best action for each state
while (delta > theta):
    sweep += 1
    delta = 0
    for i in range(1,100):# for each state
        v = Vs[i]
        #Vs[i],bestAction[i] = maxValue(i,Vs)
        Vs[i] = maxValue(i,Vs)

        delta = max(delta,abs(v - Vs[i]))
    if sweep == 1 or sweep == 2 or sweep == 3:
        y_plot[sweep-1] = Vs[1:100]
y_plot[3] = Vs[1:100]

if ph != 0.55 :
    sweep_plot(y_plot,sweep)
else:
    plt.title('ph = {}'.format(ph))
    x = np.arange(1,100)
    y = Vs[1:100]
    plt.plot(x,y)
    plt.xticks([1,25,50,75,99])
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.show()

# find optimal policy for each state
for i in range(1,100):
    action = np.arange(1,min(i,100-i)+1)
    optimal = np.zeros(action.shape) # store the value for each action

    for act in action:
        if i + act >= 100:
            # reach the goal
            optimal[act-1] = ph * (1 + Vs[act+i]) +(1-ph) * (0 + Vs[i-act])
        else:
            # normal situation
            optimal[act-1] = ph * (0 + Vs[act+i]) +(1-ph) * (0 + Vs[i-act])
    bestAction[i] = action[np.argmax(optimal)]
oplicy_plot(bestAction)
