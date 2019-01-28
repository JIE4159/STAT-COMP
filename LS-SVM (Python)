import scipy.special,scipy.linalg
import time
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_mldata
import csv
###import dataset     
charlie=pd.read_csv("D:\statistic classes\statistical computing\homework\charlie.csv",sep=',')
print(charlie)

##USing mapping method to change Data column into two classes 1 and -1
  
charlie=pd.DataFrame(data=charlie)
target={'Original':1,'New':-1}
charlie['Data']=charlie['Data'].map(target)
print(charlie)

from sklearn.model_selection import train_test_split

xvar,target=charlie.ix[:,2:6].values,charlie.ix[:,0].values



##SVM finding classifer 
import math
import numpy as np


def classifer(x,y,length,sigma,gamma):
    """ LS-SVM classifier
    Args:
        x: features matrix
        y: target matrix values of -1 and 1
        length: A integer means the numbers of train data
        sigma: A integer means kernel function's parameter
        gamma: A integer means relaxation
    Returns:
        Omega Matrix to classify observations
    """
    para=Smatrix(x[0:length],y[0:length],sigma,gamma)
    beta=para[0]
    alpha=para[1:length+1]
    result=np.zeros(len(x))
    for i in range(len(x)):
        result[i]=beta
        for j in range(length):
            result[i] +=alpha[j]*y[j]*Kernel(x[i],x[j],sigma) ##classifier function
    return result

## choose Gauss function as Kernel function to calculate the similarity between obervations

def Kernel(x,y,sigma):
    delta=x-y
    sumSquare = float(delta.dot(delta.T))
    Kerf=np.exp(-0.5*sumSquare/(sigma**2))
    return Kerf

##check kernel function by data
for i in range(len(xvar)):
    for j in range(len(xvar)):
        print(Kernel(xvar[i],xvar[j],1))
##define the function of omega in final matrix function

def Omega(x,y,sigma):
    length=len(x)
    omega=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            omega[i,j]=y[i]*y[j]*Kernel(x[i],x[j],sigma)
    return omega

##use charlie dataset to check omega function
Omega(xvar,target,1)

## solve metrix function 
##solution by inverse matrix to find A and B function
def Smatrix(x,y,sigma,gamma):
    length=len(x)+1
    ##use the YA=0 and YB+(OMEGA+C-1I)A=1 to find A and B by inverse Y
    A=np.zeros((length,length))
    A[0][0]=0
    A[0,1:length]=y.T
    A[1:length,0]=y
    A[1:length,1:length]=Omega(x,y,sigma)+np.eye(length-1)/gamma
    
    B=np.ones((length,1))
    B[0][0]=0
    ##solve inverse W
    return np.linalg.solve(A,B)

##check  Smatrix by dataset
Smatrix(xvar,target,1,2)   

##problem1 question 2 
## splitting dataset into 75% training data and 25% testing dataset
x_train, x_test, y_train, y_test = train_test_split(xvar, target, test_size=0.25, random_state=0)
##calculate model error
##training model error rate
np.sign(classifer(x_train,y_train,22,2,3))
print(np.sign(classifer(x_train,y_train,22,2,3))-y_train)
errorrates=1/22
print('The error rates of training model is %s' %errorrates)

##prediction and testing model error rates when sigma=2 gamma=3
##find beta from training model
beta1=Smatrix(x_train,y_train,2,3)[0]
alpha1=Smatrix(x_train,y_train,2,3)[1:]
gamma=3

## finding the kernel for testing dataset
##testing the distance between given x in testing dataset and x in training dataset
Kernel1=np.zeros((8,22))
for i in range(8):
    for j in range(22):
        Kernel1[i,j]=Kernel(x_test[i],x_train[j],2)
        
##check the shape of alpha1, y_train, Kernel1
np.shape(alpha1)
np.shape(Kernel1)
np.shape(y_train)
###based on the kernel function predict the sign of classifier        
predict=np.zeros(8)
for i in range(8):       ##for given x in testing dataset
    predict[i]=beta1
    for j in range(22):
        predict[i] +=alpha1[j]*y_train[j]*Kernel1[i,j]
predict         
##calculating the error rates
print(np.sign(predict)-y_test)
Trates=2/8
print('The error rates of testing dataset is %s' %Trates)


##change the parameter sigma=2 and gamma=1
np.sign(classifer(x_train,y_train,22,2,1))
print(np.sign(classifer(x_train,y_train,22,2,1))-y_train)
errorrates=2/22
print('The error rates of training model is %s' %errorrates)


##find beta from training model
beta1=Smatrix(x_train,y_train,2,1)[0]
alpha1=Smatrix(x_train,y_train,2,1)[1:]
gamma=1

## finding the kernel for testing dataset
##testing the distance between given x in testing dataset and x in training dataset
Kernel1=np.zeros((8,22))
for i in range(8):
    for j in range(22):
        Kernel1[i,j]=Kernel(x_test[i],x_train[j],2)
        
##check the shape of alpha1, y_train, Kernel1
np.shape(alpha1)
np.shape(Kernel1)
np.shape(y_train)
###based on the kernel function predict the sign of classifier        
predict=np.zeros(8)
for i in range(8):       ##for given x in testing dataset
    predict[i]=beta1
    for j in range(22):
        predict[i] +=alpha1[j]*y_train[j]*Kernel1[i,j]
predict         
##calculating the error rates
print(np.sign(predict)-y_test)
Trates=4/8
print('The error rates of testing dataset is %s' %Trates)










###problem 2 question 1 pthon for classfier of LS-OCSVM

## import the training dataset with features z1 and z2

charlie=pd.read_csv("D:\statistic classes\statistical computing\homework\charlie.csv",sep=',')

charlie=pd.DataFrame(data=charlie)
target={'Original':1,'New':-1}
charlie['Data']=charlie['Data'].map(target)
print(charlie)
##new features z1 and z2 and single value reponse orginal 
Z,original=charlie.ix[0:19,6:8].values,charlie.ix[0:19,0].values
Z_test,new_test=charlie.ix[20:30,6:8].values,charlie.ix[20:30,0].values


##The codes to solve LS-OCSVM problem.


# choose Gauss function as Kernel function to calculate the similarity between obervations

def Kernel2(x,y,sigma):
    delta=x-y
    sumSquare = float(delta.dot(delta.T))
    Kerf=np.exp(-0.5*sumSquare/(sigma**2))
    return Kerf
##compute kernel between z1 and z2 by training dataset
for i in range(len(Z)):
    for j in range(len(Z)):
        print(Kernel2(Z[i],Z[j],1))

## solve metrix function 
##solution by inverse matrix to find A and B function。
## here y are all equal to 1
def Omega2(x,sigma):
    length=len(x)
    omega2=np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            omega2[i,j]=Kernel2(x[i],x[j],sigma)
    return omega2

## solve metrix function 
##solution by inverse matrix to find A and B function

def Smatrix2(x,sigma,gamma):
    length=len(x)+1
    ##use the YA=0 and YB+(OMEGA+C-1I)A=1 to find A and B by inverse Y
    A=np.zeros((length,length))
    A[0][0]=0
    A[0,1:length]=np.ones((length-1,1)).T
    A[1:length,0]=np.ones((length-1,1))[0:length,0]
    A[1:length,1:length]=Omega2(x,sigma)+np.eye(length-1)/gamma
    
    B=np.zeros((length,1))
    B[0][0]=1
    ##solve inverse W
    return np.linalg.solve(A,B)
Smatrix2(Z,10,2)

##define the classifier function， compare to LS-SVM classifier function
 

def OCSVMclassifer(x,length,sigma,gamma):
    para=Smatrix2(x,sigma,gamma)
    beta=para[0]
    alpha=para[1:length+1]
    result=np.zeros(len(x))
    for i in range(len(x)):
        result[i]=beta
        for j in range(length):
            result[i] +=alpha[j]*Kernel2(x[i],x[j],sigma) ##classifier function
    return result


##use trainning dataset to check classfier function
ocsvm=np.sign(OCSVMclassifer(Z,20,5**(0.5),8))
error=list(ocsvm).count(-1)
errorrates=error/20
print(errorrates)

## here y=1 and the equation change slightly
##find beta and alpha from training model
beta2=Smatrix2(Z,5**(0.5),8)[0]
alpha2=Smatrix2(Z,5**(0.5),8)[1:]
gamma=8


## finding the kernel for testing dataset
##testing the distance between given x in testing dataset and x in training dataset
KernelT=np.zeros((10,20))
for i in range(10):
    for j in range(20):
        KernelT[i,j]=Kernel2(Z_test[i],Z[j],5**(0.5))
        

###based on the kernel function predict the sign of classifier        
predict=np.zeros(10)
for i in range(10):       ##for given x in testing dataset
    predict[i]=beta2
    for j in range(20):
        predict[i] +=alpha2[j]*KernelT[i,j]
predict         
##calculating the error rates
Tocsvm=np.sign(predict)
Terror=list(Tocsvm).count(1)
Terrorrates=Terror/10
print('The error rates of testing dataset is %s' %Terrorrates)

