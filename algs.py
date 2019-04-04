# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:20:42 2019

This is the home of two functions that are used by kmeans_alt.py

@author: Thomas Athey
"""

import numpy as np

"""
Cluster implements variations of Lloyd's algorithm for k means
Inputs:
    data - nxd numpy array of the data
    k - number of clusters
    iters - number of iterations
    dist - selects a certain distance metric according to different models
    
Outputs:
    cs - kxd numpy array of cluster centers
    assng - k numpy array of cluster assignments
"""
def cluster(data,k,iters,dist):
    n = data.shape[0]
    d = data.shape[1]
    cs = np.zeros((k,d))
    for i in np.arange(k):
        cs[i,:] = data[np.random.randint(n),:]
    for i in np.arange(iters):
        diffs = [data-c for c in cs]
        if dist == 'k means':
            distances = np.array([np.sum(diffs[j]**2,axis=1) for j in range(k)]).T
        elif dist == 'Unbalanced':
            if i==0:
                balance = np.ones(k)
            else:
                balance = [np.sum(assgn==j)/n for j in range(k)]
            distances = np.array([np.sum(diffs[j]**2,axis=1)*0.5-np.log(balance[j]) for j in range(k)]).T
        elif dist == 'Spheres':
            if i==0:
                vs = np.ones(k)
            else:
                vs = [np.sum(diffs[j]**2)/(d*np.sum(assgn==j)) for j in range(k)]
            distances = np.array([np.sum(diffs[j]**2,axis=1)/(2*vs[j]) + d/2*np.log(vs[j]) for j in range(k)]).T
            
        assgn = np.argmin(distances,axis=1)
        
        for i in np.arange(k):
            cs[i,:] = np.mean(data[assgn==i,:],axis=0)
            
    return cs, assgn


"""
data returns random datapoints according to different mixtures of 2 gaussians
Inputs:
    dataset - string that specifies the mixture type
        k means - balanced, spherical gaussians
        Unbalanced - unbalanced, spherical gaussians
        Spheres - balanced, differently sized spherical gaussians
    n - number of datapoints
Outputs:
    X - nxd numpy array of datapoints
    assgn - n array of true mixture assignments
"""
def data(dataset,n):
    if dataset == 'k means':
        p = 0.5
        mu0 = np.array([0,0])
        mu1 = np.array([10,0])
        s0 = np.array([[1,0],[0,1]])
        s1 = np.array([[1,0],[0,1]]) 
    elif dataset == 'Unbalanced':
        p = 0.01
        mu0 = np.array([0,0])
        mu1 = np.array([10,0])
        s0 = np.array([[1,0],[0,1]])
        s1 = np.array([[1,0],[0,1]])
    elif dataset == 'Spheres':
        p = 0.5
        mu0 = np.array([0,0])
        mu1 = np.array([10,0])
        s0 = np.array([[0.1,0],[0,0.1]])
        s1 = np.array([[5,0],[0,5]])
    
    X = np.zeros((n,2))
    assgn = np.zeros(n)
    for i in np.arange(X.shape[0]):
        u = np.random.uniform()
        if u < p:
            X[i,:] = np.random.multivariate_normal(mu0,s0)
            assgn[i] = 0
        else:
            X[i,:] = np.random.multivariate_normal(mu1,s1)
            assgn[i] = 1
    
    return X,assgn
    