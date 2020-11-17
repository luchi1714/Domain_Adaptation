# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:15:03 2020

@author: Shamprikta Mehereen and Oluchi Ibeneme
"""
import scipy.io as sio
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from numpy import random
import scipy
from sklearn.preprocessing import normalize
import  ot 
from sklearn.neighbors import KNeighborsClassifier



#Exercise 2



#dataT = sio.loadmat("C:/Users/oluch/OneDrive/Documents/UJM/M2S1/ADVANCED_ML/lab2_data/surf/dslr.mat")
#dataS = sio.loadmat("C:/Users/oluch/OneDrive/Documents/UJM/M2S1/ADVANCED_ML/lab2_data/surf/webcam.mat")


def Entropic_regularized_Optimal_transport(dataS,dataT, reg_e):
    """
    dataS: Source data
    dataT: Target data 
    reg_e    : Entropic regularization parameter
    """
    

    S = dataS["fts"]
    T = dataT["fts"]
    Ls = dataS["labels"].ravel()
    Tl = dataT["labels"].ravel()
    
    
    scaler = StandardScaler()
    S = scaler.fit_transform(S)
    T = scaler.fit_transform(T)
    
    
    
    a = np.zeros((0,S.shape[0] ))
    b = np.zeros((0,T.shape[0] ))
    
    M = scipy.spatial.distance.cdist(S, T)
    M = normalize(M, norm = "max")
    
    G = ot.sinkhorn(a, b, M, reg= 1)
    
    Sa = np.dot(G,T)
    nn = KNeighborsClassifier(n_neighbors=1).fit(Sa, Ls)
    
    #print(nn.predict(T))
    
    
    
    start = time.time()
    #Checking performance on the training set
    print('Accuracy of K-NN classifier on training set: {:.2f}'.format(nn.score(S, Ls)))
    end = time.time()
    print(" Time : {:.2f}s".format(end-start))
    
    start = time.time()
    #Checking performance on the test set
    print('Accuracy of K-NN classifier on test set: {:.2f}'.format(nn.score(T, Tl)))
    end = time.time()
    print(" Time : {:.2f}s".format(end-start))

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
        Entropic_regularized_Optimal_transport(data_1,data_2, 1)

        #print(dataS + "\n" + dataT +"\n\n")
        j = j +1








