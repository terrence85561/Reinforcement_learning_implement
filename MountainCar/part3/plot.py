import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

steps = 50
z = np.load('neg_q_hat.npy')

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

x = np.arange(-1.2,0.5,(0.5+1.2)/steps)
y = np.arange(-0.07,0.07,(0.07+0.07)/steps)
xx,yy = np.meshgrid(x,y)

ax.set_xticks([-1.2,0.5])
ax.set_yticks([-0.07,0.07])
ax.set_zticks([0,np.max(z)])
ax.set_xlabel('position')
ax.set_ylabel('velocity')


ax.plot_surface(xx, yy, z, rstride=2, cstride=2, cmap=plt.get_cmap('rainbow'))

plt.show()


