#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568

import sys
import numpy as np
import pandas as pd
from numpy import linalg as LA

def main():
    data = pd.read_csv(sys.argv[1],sep=',',header=None)
    print(Kmeans(data,3))

def Kmeans(data,k):
    centroids = data.sample(k)
    x = data[0].values
    y = data[1].values 
    Pdata = np.array(list(zip(x,y)))
    C_x = centroids[0].values
    C_y = centroids[1].values
    Cdata = np.array(list(zip(C_x,C_y)))
    oldCentroids = Cdata.copy()
    while not Termination(oldCentroids, centroids):
        oldCentroids = centroids
        #Calculating Labels
        labels = getLabels(Pdata,Cdata)
        #Calculating Centroids
        centroids = getCentroids(Pdata, labels, k, Cdata)
        
    return centroids

def Termination(oldCentroids, centroids):
    if (distance(oldCentroids, centroids,ax=0).all()>0.001): return True
    return False

def getLabels(Pdata,Cdata):
    
    labels = np.zeros(len(Pdata))
    for i in range(len(Pdata)):
        distances = distance(Pdata[i],Cdata, ax=1)
        label = np.argmin(distances)
        labels[i] = label
    
    return labels

def getCentroids(Pdata, labels, k, Cdata):
    for i in range(k):
        points = [Pdata[j] for j in range(len(Pdata)) if labels[j]==i]
        Cdata[i] = np.mean(points, axis=0)   
    return Cdata

def distance(a, b, ax):
    return LA.norm(a-b,axis=ax)

if __name__ == '__main__':
    main()
