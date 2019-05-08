import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load('output.npy')
    

    

    #for arr in data:
    x = np.arange(0,50)
    
    plt.plot(x, data[0], label='q-learning')
    plt.plot(x, data[1], label='n=5')
    plt.plot(x, data[2], label='n=50')

    plt.xlim([0, 50])
    #plt.xticks(np.arange(0, 2000))
    plt.xlabel('Episodes')
    plt.ylabel('steps per episode')
    plt.legend()
    plt.show()
