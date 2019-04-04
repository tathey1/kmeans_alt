# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:49:58 2019

Compares two clustering methods according to ARI

Uses functions from algs.py

@author: Thomas Athey
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from algs import cluster,data
from sklearn.metrics import adjusted_rand_score

n = 5000
dataset = 'Unbalanced' #possibilities - k means, Unbalanced, Spheres
alg1 = 'k means' #same possibilities as the dataset possibilities
alg2 = dataset

iters = 10 #number of simulations
aris = np.zeros((iters,2)) #matrix to store results

#this loop generates random data, and clusters according to both algorithms
#then compares ARI results
for i in range(iters):
    
    if i%int(iters/10) == 0: #prints progress
        print(str(int(100*i/iters))+'%')
        
    X,assgn_true = data(dataset,n) #generate data

    #first clustering
    cs, assgn = cluster(X,2,10,alg1) 
    #in case you want to plot the data and the clustering
    #plt.figure(figsize=(8,8))
    #plt.scatter(X[:,0],X[:,1],c=assgn)
    aris[i,0] = adjusted_rand_score(assgn_true,assgn)
    
    #second clustering
    cs, assgn = cluster(X,2,10,alg2) 
    #plt.figure(figsize=(8,8))
    #plt.scatter(X[:,0],X[:,1],c=assgn)
    aris[i,1] = adjusted_rand_score(assgn_true,assgn)


#the rest of the code overlays the histograms and creates a break in the x axis
bins = np.linspace(-0.05,1,21)
f, (ax1,ax2) = plt.subplots(1,2,sharey=True,facecolor='w')
ax1.hist(aris[:,0],bins,alpha=0.5,label=alg1)
ax1.hist(aris[:,1],bins,alpha=0.5,label=alg2)
ax2.hist(aris[:,0],bins,alpha=0.5,label=alg1)
ax2.hist(aris[:,1],bins,alpha=0.5,label=alg2)

ax1.set_xlim(-0.05,0.1)
ax2.set_xlim(0.9,1)

ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.yaxis.tick_left()
ax1.tick_params(labelright='off')
ax2.yaxis.tick_right()

ax1.set_xticks(np.arange(-0.05,0.1,0.05))
ax2.set_xticks(np.arange(0.9,1.01,0.05))

d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((1-d,1+d), (-d,+d), **kwargs)
ax1.plot((1-d,1+d),(1-d,1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d,+d), (1-d,1+d), **kwargs)
ax2.plot((-d,+d), (-d,+d), **kwargs)

ax1.set_ylabel('Count')
ax1.set_xlabel('ARI')
ax2.legend(loc='upper right')

#f.savefig('C:/Users/Thomas Athey/Documents/Labs/Labs/jovo/clustering/gmmvskmeans/img/unbalanced.jpg')