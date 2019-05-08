import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load('windy.npz')
    print("time steps",data['timeSteps'])
    print("Episodes",data["Episodes"])

    y_episode = data["Episodes"]
    x_timeSteps = data["timeSteps"]

    plt.plot(x_timeSteps,y_episode)
    plt.xlim([0,8000])
    plt.xticks(np.arange(0,8001,step = 1000))
    plt.xlabel('Times Steps')
    plt.ylabel('Episodes')
    plt.show()
