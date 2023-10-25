import numpy as np
import matplotlib.pyplot as plt
import bellman as bel
import pm
from simProj import euclidean_proj_simplex as Proj

mode = 'exact'
tol = 0.0001
N = 3000
RReturn= []
def polUpdate(pi,gr,eta=0.001):
    for s in range(pm.S):
        pi[s] = Proj(pi[s]+eta*gr[s]) 
    return pi

######################################
rect ='s'
p=2

pi = np.ones((pm.S,pm.A))/pm.A 
pm.pi = pi
Return = []
for i in range(N):
    gr = bel.RPG(rect=rect,p=p)
    pi = polUpdate(pm.pi,gr)
    v = pm.v0
    iter = int(np.log(pm.tol/(pm.S*pm.A))/np.log(pm.gamma))
    for j in range(iter):   # value function approximation 
        v =bel.RVI(v=v,p=p,rect=rect,mode=mode,tol=tol)
        r = np.mean(v)
    Return.append(r)
    print(i)
    pm.pi = pi
plt.plot(Return)
RReturn.append(Return)

######################################

rect ='sa'
p=2
pi = np.ones((pm.S,pm.A))/pm.A 
pm.pi = pi
Return = []
for i in range(N):
    gr = bel.RPG(rect=rect,p=p)
    pi = polUpdate(pm.pi,gr)
    v = pm.v0
    iter = int(np.log(pm.tol/(pm.S*pm.A))/np.log(pm.gamma))
    for j in range(iter):   # value function approximation 
        v =bel.RVI(v=v,p=p,rect=rect,mode=mode,tol=tol)
        r = np.mean(v)
    Return.append(r)
    print(i)
    pm.pi = pi
plt.plot(Return)
RReturn.append(Return)

############################################

rect='nr'
pi = np.ones((pm.S,pm.A))/pm.A 
pm.pi = pi
Return = []
for i in range(N):
    gr = bel.RPG(rect=rect,p=p)
    pi = polUpdate(pm.pi,gr)
    v = pm.v0
    iter = int(np.log(pm.tol/(pm.S*pm.A))/np.log(pm.gamma))
    for j in range(iter):   # value function approximation 
        v =bel.RVI(v=v,p=p,rect=rect,mode=mode,tol=tol)
        r = np.mean(v)
    Return.append(r)
    print(i)
    pm.pi = pi

RReturn.append(Return)
plt.plot(Return)
########################################################

plt.legend(['U^s_2','U^sa_2','nr'])
plt.xlabel('RPG iterations')
plt.ylabel('Robust Return')
plt.title('Convergence of Robust Policy Gradient Method')
plt.savefig('RPG_methods_S{}A{}.png'.format(pm.S,pm.A))
np.savetxt('RPG_methods_S{}A{}.txt'.format(pm.S,pm.A),RReturn)