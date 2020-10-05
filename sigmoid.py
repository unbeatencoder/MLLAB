#This file contains code for sigmoid function.
import numpy as np 
def sig(z):
    """
    sigmoid activation function.

    inputs: z
    outputs: sigmoid(z)
    """
    s = 1. / (1. + np.exp(-z))
    return s
