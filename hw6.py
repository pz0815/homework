# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]

# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],[0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],[0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# m1: mean of class 1
# m2: mean of class 2
m1 = np.mean(X1, axis=0, keepdims=1)
m2 = np.mean(X2, axis=0, keepdims=1)

cov1 = np.cov(X1.T)  
cov2 = np.cov(X2.T)  

# Step 2
eigvals1, eigvecs1 = myeig(cov1, symmetric=True)
eigvals2, eigvecs2 = myeig(cov2, symmetric=True)

# Step 3
plt.figure(dpi=288)

plt.plot(X1[:, 0], X1[:, 1], 'r.', label='Class 1')
plt.plot(X2[:, 0], X2[:, 1], 'g.', label='Class 2')

for i in range(2):
    plt.arrow(m1[0, 0], m1[0, 1],
              eigvecs1[0, i] * np.sqrt(eigvals1[i]) * 2,
              eigvecs1[1, i] * np.sqrt(eigvals1[i]) * 2,
              head_width=0.2, color='red', alpha=0.5)

for i in range(2):
    plt.arrow(m2[0, 0], m2[0, 1],
              eigvecs2[0, i] * np.sqrt(eigvals2[i]) * 2,
              eigvecs2[1, i] * np.sqrt(eigvals2[i]) * 2,
              head_width=0.2, color='green', alpha=0.5)

plt.title('Class 1 and Class 2 with Principal Axes')
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()
