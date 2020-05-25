import numpy as np 
import pandas as pd 
import scipy.optimize as opt
from CostFunction import CostFunc
from Gradients import Grad
def onevsAllAlgo(data_x,data_y,num_labels, lam):
    m = data_x.shape[0]
    n = data_x.shape[1]
    finalTheta = np.zeros((num_labels, n + 1))
    data_x = np.c_[np.ones((m,1)),data_x]
    initial_theta = np.zeros(n+1)
    new_theta = initial_theta
    # theta = opt.fmin_cg(CostFunc, fprime= Grad,
    #             x0=initial_theta, args=(data_x, data_y, lam), maxiter=50)
    # print(theta)
    iter = 100
    previouserror = -10000000
    currenterror = 0
    while(previouserror<currenterror):
        iter=iter - 1
        previouserror = CostFunc(initial_theta,data_x,data_y,lam)
        print(previouserror)
        initial_theta = new_theta
        grad = Grad(initial_theta,data_x,data_y,lam)
        new_theta = initial_theta - 0.001*grad
        currenterror = CostFunc(new_theta,data_x,data_y,lam)
        print(currenterror)

    print(CostFunc(initial_theta,data_x,data_y,lam))
    return initial_theta