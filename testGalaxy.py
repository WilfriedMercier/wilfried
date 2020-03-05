import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from wilfried.galaxy import mergeModelsIntoOne

f = plt.figure(figsize=(21, 7))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

x    = np.arange(0, 101, 1)
y    = x
X, Y = np.meshgrid(x, y)
Z    = X.copy()*0+1
Z2   = -Z
X2   = X.copy()+30

X3, Y3, Z3 = mergeModelsIntoOne([X, X2], [Y, Y], [Z, -Z], 1, 1)

contours = [-1, -0.5, 0.5, 1]
cmap     = 'bwr'
ax1.contourf(X, Y, Z, levels=contours, cmap=cmap)
ax2.contourf(X2, Y, Z, levels=contours, cmap=cmap)
data = ax3.contourf(X3, Y3, Z3, levels=contours, cmap=cmap)
plt.colorbar(data)
plt.show()
