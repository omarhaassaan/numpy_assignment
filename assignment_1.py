import numpy as np

def randomization(n):
    return np.random.random((n,1))

def operations(h, w):
    a = np.random.random(size=(h,w))
    b = np.random.random(size=(h,w))
    return (a,b,a+b)

def norm(A, B):
    return np.linalg.norm(A+B)

def neural_network(inputs, weights):
    import math
    return math.tanh(np.matmul(weights.transpose(),inputs))

def scalar_function(x, y):
    if(y>x):
        return x/y
    return x*y

def vector_function(x, y):
    func = np.vectorize(scalar_function)
    return func(x,y)


