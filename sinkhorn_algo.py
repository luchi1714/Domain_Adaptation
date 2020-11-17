# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:15:03 2020

@author: Shamprikta Mehereen and Oluchi Ibeneme

"""


import scipy.io as sio
import time
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier



def sinkhorn_knopp(dataS,dataT, reg_e = 1, iters = 100, tolerance = 1e-5):
     """
    dataS: Source data
    dataT: Target data 
    reg_e: Entropic regularization parameter
    iters: Number of iterations
    tolerance : Sensitivity to change 
    """
    
    S = dataS["fts"]
    T = dataT["fts"]
    Ls = dataS["labels"]
    Ls = Ls.ravel()
    Tl = dataT["labels"]
    Tl = Tl.ravel()
    
    
    M = cdist(S, T)
    row, column = M.shape
    
    a = np.ones(row)
    b = np.ones(column)
    
    
    
    # VECTORS
    u = np.ones(row)
    v = np.ones(column)

    
    K = np.exp(-M/reg_e)
    x = np.empty(b.shape, dtype=M.dtype)
    Kp = (1 / a).reshape(-1, 1) * K
    
    
    epsilon = 1
    
    
    #Iterate until convergence 
    
    for ite in range(iters):
        if (epsilon > tolerance):
            
            #Update vectors
            
            v = np.divide(b, K.T.dot(u))
            u = 1. / np.dot(Kp, v)
            
            


            x = u.dot(K.dot(v))
            epsilon = np.linalg.norm(x - b)
          
            
            
    G = u.reshape((-1, 1)) * K * v.reshape((1, -1))
    S_a = G.dot(T)
    start = time.time()
    classifier = KNeighborsClassifier(n_neighbors = 1)
    classifier.fit(S_a, Ls)
    ypred = classifier.predict(T)
    end  = time.time()
    accuracy = metrics.accuracy_score(Tl, ypred)
    print("Accuracy: { :.2f}".)
    print("{:.2f}s".format(end-start))


# =============================================================================
#         LOADING DOCUMENTS PATH
# =============================================================================

folders = ["surf", "GoogleNet1024", "CaffeNet4096"]
files = ["webcam", "dslr", "caltech10", "amazon"]

# enter your path here 
directory_path = "C:/Users/oluch/OneDrive/Documents/UJM/M2S1/ADVANCED_ML/lab2_data/Data_Folder"


# =============================================================================
#         GENERATING SOURCE-TARGET COMBINATIONS FROM FOLDERS IN PATH
# =============================================================================
for i in folders:
    j = 0
    print(" \t\t\t\t\t\t{}                      ".format(i.capitalize()))
    while(j < 3):
        dataS = (directory_path+"/{}/{}.mat".format(i,files[j]))
        j = j +1
        dataT = (directory_path+"/{}/{}.mat".format(i,files[j]))
        
        
        print("*" *60)
       
        print(files[j-1].capitalize() +" vs "+files[j].capitalize()+ " \n")
       
        data_1 = sio.loadmat("{}".format(dataS))
        data_2 = sio.loadmat("{}".format(dataT))
        sinkhorn_knopp(data_1,data_2, 1,  iters = 100, tolerance = 1e-5)

        #print(dataS + "\n" + dataT +"\n\n")
        j = j +1











    
    
    
    
    
    
    
    