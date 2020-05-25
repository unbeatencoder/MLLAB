import numpy as np  
import pandas as pd
from sigmoid import sig

def Grad(theta,X,y,reg):
    #  print(X.shape)
    #  print(y.shape)
    
    #  print(theta.shape)
     
     m = X.shape[0]
     thetad = np.zeros((X.shape[1],1))
     thetad[:,0] = theta
    #  print(thetad.shape)
     y_pred = sig(np.dot(X,thetad))
    #  print(y_pred.shape)
     diff = y_pred - y
     finalGradient = np.dot(X.T,diff)
     finalGradient = finalGradient/m
     GradientDash = (thetad * reg) / m
     GradientDash[0]=0
     finalGradientRegularization = finalGradient + GradientDash
     returnValue = np.zeros(theta.shape[0])
     returnValue = finalGradientRegularization[:,0]
     return returnValue