from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

hw8_csv = pd.read_csv(r'C:\ddd\data\hw8.csv')
hw8_dataset = hw8_csv.to_numpy(dtype=np.float64)

X0 = hw8_dataset[:, 0:2]
y = hw8_dataset[:, 2]

X = np.hstack((np.ones((X0.shape[0], 1)), X0))
w = la.pinv(X.T @ X) @ X.T @ y

def decision_boundary(x1, w):
    return -(w[0] + w[1] * x1) / w[2]

fig = plt.figure(dpi=288)
plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')

x1_vals = np.linspace(X0[:, 0].min(), X0[:, 0].max(), 100)
x2_vals = decision_boundary(x1_vals, w)
plt.plot(x1_vals, x2_vals, 'k-', linewidth=2, label='Decision Boundary')

xx, yy = np.meshgrid(np.linspace(X0[:, 0].min(), X0[:, 0].max(), 500),
                     np.linspace(X0[:, 1].min(), X0[:, 1].max(), 500))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_with_bias = np.hstack((np.ones((grid.shape[0], 1)), grid))
Z = np.sign(grid_with_bias @ w).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.2, levels=[-1, 0, 1], colors=['blue', 'red'])

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.legend()
plt.title('Linear Classifier with Decision Boundary')
plt.show()
