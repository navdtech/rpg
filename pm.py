import ssl
import numpy as np
import matplotlib.pyplot as plt

# parameters
def kernel(S,A):            # Generates random transition kernel
    p = np.random.rand(S,A,S)
    for s in range(S):
        for a in range(A): 
            summ = np.sum(p[s,a])
            p[s,a] = p[s,a]/summ
    return p

def policy(S,A):                     #  Generates random policy
    p = np.random.rand(S,A)
    for s in range(S): 
            summ = np.sum(p[s])
            p[s] = p[s]/summ
    return p

tol=0.0001
S = 50
A = 20
R = np.random.randn(S,A)
v0 = np.random.randn(S)
gamma = 0.9
alphaSA = 0.1*np.ones((S,A))
betaSA = (0.01/S)*np.ones((S,A))
alphaS = 0.1*np.ones(S)
betaS = (0.01/S)*np.ones(S)
n =1
P = kernel(S,A)
Q0 = R + gamma*kernel(S,A)@v0
pi = policy(S,A)
mu = np.ones(S)/S
Gpi = np.random.randn(S,A)  # gradient of pi  \nabla pi
class pm:
    def __init__(self,S,A):
        self.S =S 
        self.A = A
        self.R = np.random.randn(S,A)
        self.v0 = v0
        self.gamma = 0.9
        self.alphaSA = 0.1*np.ones((S,A))
        self.betaSA = 0.1*np.ones((S,A))
        self.alphaS = 0.1*np.ones(S)
        self.betaS = 0.1*np.ones(S)
        self.n =100
        self.P = kernel(S,A)
        self.Q0 = R + gamma*kernel(S,A)@v0
        self.pi = policy(S,A)
        self.mu = np.ones(S)/S
        self.Gpi = np.random.randn(S,A) 


