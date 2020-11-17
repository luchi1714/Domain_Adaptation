# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 14:15:03 2020

@author: Shamprikta Mehereen and Oluchi Ibeneme
"""
import scipy.io as sio
import timeit

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier




def subspace_alignment(dataS,dataT, pca_components):
    """
    dataS: Source data
    dataT: Target data 
    d    : PCA Components
    """


    pca_components = 100
    S = dataS["fts"].astype(float)
    T = dataT["fts"].astype(float)
    Sl = dataS["labels"].ravel()
    Tl = dataT["labels"].ravel()
    
    scaler = StandardScaler()
    S = scaler.fit_transform(S)
    T = scaler.fit_transform(T)
     
    
    
    # =============================================================================
    # 
    #             Exercise 1 : Subspace alignment
    # 
    # =============================================================================
    pca = PCA(pca_components)
    
    
    pca_Xs = pca.fit(S)
    Xs =np.transpose(pca_Xs.components_)[:, :pca_components] #pca.components_
    
    
    pca_Xt = pca.fit(T)
    Xt = np.transpose(pca_Xt.components_)[:, :pca_components]
     
    
    M = np.dot(Xs, Xs.T)
    
    
    Xa = np.dot(M, Xt)
    Sa = np.dot(S, Xa)
    Ta = np.dot(T, Xt)
    Sl = np.ravel(Sl)
    nn = KNeighborsClassifier(n_neighbors=1).fit(Sa, Sl)
    
    
    start = timeit.timeit()

    predection  = nn.predict(Ta)
     
    #Checking performance on the Source set
    print('Accuracy of subspace_alignment : {:.2f}'.format(metrics.accuracy_score(Tl, predection)))
    end = timeit.timeit()
    print(" Time elapsed : {}" .format(end - start))
    #Checking performance on the Tregular knn
    
    
    start = timeit.timeit()
    nn = KNeighborsClassifier(n_neighbors=1).fit(S, Sl)
    predection_nn  = nn.predict(T)
    print('Accuracy of K-NN: {:.2f}'.format(metrics.accuracy_score(Tl,  predection_nn )))
    end = timeit.timeit()
    print(" Time elapsed : {}" .format( end - start))


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
        subspace_alignment(data_1,data_2, 105)

        #print(dataS + "\n" + dataT +"\n\n")
        j = j +1
























