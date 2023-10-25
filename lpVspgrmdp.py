import numpy as np
from bellman import RVI
import pm 
import time 

############# printing paratmeters ######
print('\n\n state={},action={}, discount factor={}, uncertainty radius={},number of iterations ={}\n\n'.format(pm.S,pm.A,pm.gamma, pm.alphaS[0],pm.n))

############ Helper #############
def Occupation(mu = pm.mu, P=pm.P, pi = pm.pi, iter=100):  # Computes occupation measure, w.r.t. initial vector mu
    p = Ppi(pm)
    k = mu 
    K = []
    for i in range(iter):
        K.append(k)
        k = pm.gamma*k@p
    return np.sum(K, axis=0)
def Ppi(pm=pm): # Computes P^{\pi}
   return np.array( [ pm.pi[i]@pm.P[i] for i in range(pm.S)])





###### Non-robust MDP ########
iter = int(np.log(pm.tol/(pm.S*pm.A))/np.log(pm.gamma))
def NR():
    v = pm.v0
    for  i in range(iter):
        v = RVI(v,pm=pm, rect='nr')

    Q = pm.R + pm.P@v
    dmu = Occupation(iter=iter)
    pg =  np.sum(np.multiply(Q,dmu[:,None])*pm.Gpi)
# st = time.time()
Temp = []
for i in range(100):
    st = time.time()
    NR()
    et = time.time()
    Temp.append(et-st)
nrt = np.mean(Temp)
nrv = np.std(Temp)
# print('number of iteration {}'.format(iter))

print('non-robust cost per iteration = {}, relative cost = 1, relative variance {}\n'.format(nrt,nrv/nrt))



# ########  Sa rectangular L_inf robust MDP by Linear Programming ###########3
# n=10
# v = pm.v0
# st1 = time.time()
# for  i in range(n):
#     v = RVI(v,pm=pm, rect='sa', mode='lp', p='inf')
#     # print(i)
# et1 = time.time()

# lpit = (et1-st1)/n
# print('number of iteration {}'.format(n))

# print('SA_inf by Linear Programming ; cost per iteration = {}, relative cost = {}\n\n\n'.format(lpit,lpit/nrt))





########  Sa rectangular L_1 robust MDP by Linear Programming ###########3
def RPGsa1():
    v = pm.v0
    for  i in range(iter):
        v = RVI(v,pm=pm, rect ='sa', mode='lp', p=1)
        # print(i)
    Q = pm.R + pm.P@v
    dmu = Occupation(iter=iter)
    pg =  np.sum(np.multiply(Q,dmu[:,None])*pm.Gpi)

# st = time.time()
# RPGsa1()
# et = time.time()
# lp1t = (et-st)
# print('number of iteration {}'.format(n))

# print('SA_1 by Linear Programming :cost per iteration = {}, relative cost = {}\n\n\n'.format(lp1t,lp1t/nrt))


# # print('Summary: realtive cost w.r.t. non-robust: [nr, LPsa1, LPsainf] ={} '.format(list(np.array([nrt,lp1t,lpit])/nrt)))




########  S rectangular L_1 robust MDP by Linear Programming ###########3
def RPGs1():
    v = pm.v0
    for  i in range(iter):
        v = RVI(v,pm=pm, rect ='s', mode='lp', p=1)
        # print(i)
    Q = pm.R + pm.P@v
    dmu = Occupation(iter=iter)
    pg =  np.sum(np.multiply(Q,dmu[:,None])*pm.Gpi)
# st = time.time()
# RPGs1()
# et = time.time()
# lps1t = (et-st)
# print('number of iteration {}'.format(n))


######   Excution time sa rectangular MDPs  ####
N=1
T = [nrt]
V =[nrv]
for fun in [RPGsa1,RPGs1]:

    Temp = []
    for n in range(N):
        start_time = time.time()
        fun()
        end_time = time.time()
        Temp.append(end_time-start_time)
        print((end_time-start_time)/nrt)
    Temp = np.array(Temp)
    T.append(np.mean(Temp))
    V.append(np.std(Temp))
print('Total time taken by sa-rectangular L_1 MDP  is {},  relative cost {} , relative std {} \n'.format( T[1], T[1]/nrt,V[1]/nrt))
print('Total time taken by s-rectangular L_1 MDP  is {},  relative cost {} , relative std {} \n'.format( T[2], T[2]/nrt,V[2]/nrt))
    
T = np.array(T)
V = np.array(V)

np.savetxt('TimelpVspgrmdpA{}A{}N{}.txt'.format(pm.S,pm.A,N),T)

np.savetxt('VariancelpVspgrmdpA{}A{}N{}.txt'.format(pm.S,pm.A,N),V)

# print('S_1 by Linear Programming :cost per iteration = {}, relative cost = {}\n\n\n'.format(lps1t,lps1t/nrt))


# print('Summary: realtive cost w.r.t. non-robust: [nr, LPs1] ={} '.format(list(np.array([nrt,lps1t])/nrt)))

print('\n\n NEW EXPERIMENT \n\n')


