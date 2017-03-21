# -*- coding: utf-8 -*-
"""     
Copyright (C) 2016  Nicola Dileo
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>


Module: imgage_clustering.py
--------
"""


import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import KMeans


def classic(image, n_clusters):
    pass

def incremental(image, n_clusters):
    pass

def fuzzy(image, n_clusters):
    pass


def fuzzy_clusterize(image, n_clust):
    height, width, r = image.shape
    size = height*width
    flattened = image.reshape((size, r))    
    new_image = np.zeros((size, r))
    
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(flattened.T, n_clust, 2, error=0.01, maxiter=1000, init=None)
    u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(flattened.T, cntr, 2, error=0.01, maxiter=1000)
    membership = np.argmax(u,axis=0)
    
    for i in range(n_clust):
        new_image[membership == i] = cntr[i]

    new_image.shape = (height,width,r)  
    image.shape = (height,width,r)
    return new_image, cntr
    
    
def km_clusterize(image, n_clust):
    height, width, r = image.shape
    size = height*width
    flattened = image.reshape((size, r))         
    new_image = np.zeros((size, r))    
            
    km = KMeans(n_clusters = n_clust,init='random',n_init=1)
    km.fit(flattened)

    for cl_label in range(n_clust):
        centroid = km.cluster_centers_[cl_label]
        new_image[np.where(km.labels_ == cl_label)] = centroid
        
    new_image.shape = (height,width,r)     
    image.shape = (height,width,r)
    return new_image, km


def colorize(image, kmeans):
    height, width, r = image.shape
    size = height*width
    flattened = image.reshape((size, r))  
    new_image = np.zeros((size, r))   
    n_clusters = kmeans.n_clusters
    predictions = kmeans.predict(flattened)
        
    for cl_label in range(n_clusters):         
        centroid = kmeans.cluster_centers_[cl_label]
        new_image[np.where(predictions == cl_label)] = centroid               
    
    new_image.shape = (height,width,r)
    return new_image


def empty_list(size):
    return [0 for i in range(0, size)]    
    
