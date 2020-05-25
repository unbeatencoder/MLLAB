import numpy as np  
import pandas as pd
from sigmoid import sig

def CostFunc(theta,X,y,reg):
    # print(theta.shape)
    # print(X.shape)
    # print(y.shape)
    
    
    thetad = np.zeros((X.shape[1],1))
    thetad[:,0] = theta
    y_pred = sig(np.dot(X,thetad))
    m = y.shape[0]
    # print(y_pred)
    y_predsubtracted1 = np.ones((m,1)) - y_pred
    # print(y_predsubtracted1)
    ysubtracted1 = np.ones((m,1)) - y
    ynegative = -y
    first_term = np.multiply(ynegative,np.log(y_pred))
    second_term = np.multiply(ysubtracted1,np.log(y_predsubtracted1))
    finalCost = np.sum(first_term-second_term)
    finalCost = finalCost/m
    thetaSquare = np.square(thetad[1:])
    finalCostRegularization = (np.sum(thetaSquare)*reg)/(2*m)
    finalCostRegularization = finalCostRegularization + finalCost
    returnCost = np.zeros(1)
    returnCost = finalCostRegularization
    return returnCost