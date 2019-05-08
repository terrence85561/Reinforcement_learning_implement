import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import csv

def plot():
    name1 = input("enter first file name with out '.csv': ")
    name2 = input("enter first file name with out '.csv': ")
    Q10y = []

    Q15y = []
    x = np.arange(1000)
    if name1:
        with open(name1+'.csv','r') as csvfile:
            plots = csv.reader(csvfile)
            for row in plots:
                Q10y.append(float(row[0]))
        plt.plot(x,Q10y,label = "xx!")
    if name2:
        with open(name2+'.csv','r') as csvfile:
            plots = csv.reader(csvfile)
            for row in plots:
                Q15y.append(float(row[0]))
        plt.plot(x,Q15y,label = "xxxx!")


    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xlabel('steps')
    plt.ylabel('%'+'\n'+'Optimal\naction')
    plt.ylim((0.0,1.0))

    plt.show()
def plot_ucb():
    name = input("enter file name with out '.csv': ")
    q_ucb = []
    with open(name+'.csv','r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            q_ucb.append(float(row[0]))
    x = np.arange(1000)
    plt.plot(x,q_ucb,label = "ucb")

    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.xlabel('steps')
    plt.ylabel('%'+'\n'+'Optimal\naction')
    plt.ylim((0.0,1.0))
    plt.show()


def to_percent(temp,position):
    return '%1.0f'%(100*temp) + '%'



def main():
    while(True):
        choice = input("Enter 'U' to plot ucb graph, enter 'E' to plot the epsilon-greedy and optimal initial value graph,\nenter 'q' to quit.\nYour answer:\t")
        if choice == "U":
            plot_ucb()
        if choice == "E":
            plot()
        if choice == 'q':
            return
if __name__ == '__main__':
    main()
