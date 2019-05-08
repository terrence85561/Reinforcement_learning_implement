import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = np.load('randomwalk.npy')
    print("tabular", data[0])
    print("tile coding", data[1])

    #for arr in data:
    x = np.arange(0,2000,10)
    
    plt.plot(x, data[0], label='tabular')
    plt.plot(x, data[1], label='tile_coding')

    plt.xlim([0, 2000])
    #plt.xticks(np.arange(0, 2000))
    plt.xlabel('Episodes')
    plt.ylabel('RMSE on average 30 runs')
    plt.legend()
    plt.show()
