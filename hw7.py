# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

# If this script is not run under spyder IDE, comment the following two lines.
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

def scatter_pts_2d(x, y):
    # set plotting limits
    xmax = np.max(x)
    xmin = np.min(x)
    xgap = (xmax - xmin) * 0.2
    xmin -= xgap
    xmax += xgap

    ymax = np.max(y)
    ymin = np.min(y)
    ygap = (ymax - ymin) * 0.2
    ymin -= ygap
    ymax += ygap 

    return xmin,xmax,ymin,ymax

dataset = pd.read_csv('data/hw7.csv').to_numpy(dtype = np.float64)
x = dataset[:, 0]
y = dataset[:, 1]

# parameters for our two runs of gradient descent
w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])

alpha = 0.05
max_iters = 500

for _ in range(1, max_iters):
    w0_grad = -2 * np.sum(y - w[0] - w[1] * np.sin(w[2] * x + w[3]))
    w1_grad = -2 * np.sum((y - w[0] - w[1] * np.sin(w[2] * x + w[3])) * np.sin(w[2] * x + w[3]))
    w2_grad = -2 * np.sum((y - w[0] - w[1] * np.sin(w[2] * x + w[3])) * w[1] * x * np.cos(w[2] * x + w[3]))
    w3_grad = -2 * np.sum((y - w[0] - w[1] * np.sin(w[2] * x + w[3])) * w[1] * np.cos(w[2] * x + w[3]))
    
epsilon = 1e-6  
for _ in range(1, max_iters):
    gradient = np.zeros_like(w)
    for i in range(len(w)):
        w_temp1 = w.copy()
        w_temp2 = w.copy()
        w_temp1[i] += epsilon
        w_temp2[i] -= epsilon
        
        cost1 = np.sum((y - w_temp1[0] - w_temp1[1] * np.sin(w_temp1[2] * x + w_temp1[3]))**2)
        cost2 = np.sum((y - w_temp2[0] - w_temp2[1] * np.sin(w_temp2[2] * x + w_temp2[3]))**2)
        
        gradient[i] = (cost1 - cost2) / (2 * epsilon)
    
    w -= alpha * gradient



xmin,xmax,ymin,ymax = scatter_pts_2d(x, y)
xt = np.linspace(xmin, xmax, 100)
yt1 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

w = np.array([-0.1607108,  2.0808538,  0.3277537, -1.5511576])
for _ in range(1, max_iters):
    pass
    # remove the above pass and write your code here
    # calculate gradient of cost function by using numeric method(使用數值法計算梯度)
    # update rule: 
    #     w =  w - alpha * gradient_of_cost
    

xt = np.linspace(xmin, xmax, 100)
yt2 = w[0] + w[1] * np.sin(w[2] * xt + w[3])

# plot x vs y; xt vs yt1; xt vs yt2 
fig = plt.figure(dpi=288)
plt.scatter(x, y, color='k', edgecolor='w', linewidth=0.9, s=60, zorder=3)
plt.plot(xt, yt1, linewidth=4, c='b', zorder=0, label='Analytic method')
plt.plot(xt, yt2, linewidth=2, c='r', zorder=0, label='Numeric method')
plt.xlim([xmin,xmax])
plt.ylim([ymin,ymax])
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()
