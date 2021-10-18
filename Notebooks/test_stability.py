# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:26:43 2020

@author: krs234
"""


import numpy as np
import matplotlib.pyplot as plt


# Number of components
N = 50;

# Distribution of interactions (mean,variance)
mean = -0.0;
std = 0.4;
chi = std*np.random.randn(N,N) + mean;
np.fill_diagonal(chi, 0);
for i in np.arange(len(chi)-1):
    for j in np.arange(i+1,len(chi)):
        chi[i,j] = chi[j,i]
        
# Define initial concentrations
phi = np.ones((N,1))/N;
r_mu = 20;
r_sigma = 0.0;
r = r_sigma*np.random.randn(N,1) +r_mu ;
r[r<1] = 1;
# Define Jacobian matrix
D = np.zeros((N,N));
np.fill_diagonal(D, 1/(phi*r))

J = D + chi ;
w, v = np.linalg.eig(J)

fig,axes = plt.subplots(2,1)
axes[0].hist(w)
axes[0].text(0.8,0.8,"min(eig) = " + str(round(np.min(w),2)),transform=axes[0].transAxes)

axes[1].plot(v[:,np.where(w==np.min(w))[0][0]])
axes[1].text(0.5,0.1,"Positive directions = " + str(sum(v[:,np.where(w==np.min(w))[0][0]]>0)),transform=axes[1].transAxes)
plt.show()