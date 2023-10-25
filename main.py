from math import gamma
import numpy as np
import matplotlib.pyplot as plt
from bellman import RPG
import time 
import pm 
############# printing paratmeters ######
print('\n\n state={},action={}, discount factor={}, uncertainty radius={},number of iterations ={}\n\n'.format(pm.S,pm.A,pm.gamma, pm.alphaS[0],pm.n))

#### Paramaters
N = 100
tol = 0.0001



####  Non Robust Policy Gradient  #####
print('Evaluation of non-robust starts for time measurement','\n')

start_time = time.time()
for n in range(N):
    pg = RPG(rect='nr')
end_time = time.time()
nrt = (end_time-start_time)/N
print('Total time taken by non-robust MDP  is {}, relative cost 1 \n'.format( nrt))




######   Excution time sa rectangular MDPs  ####
Tsa = []

for p in [1,2,3,'inf']:

    start_time = time.time()
    for n in range(N):
        pg = RPG(p=p,rect='sa',tol=tol)
    end_time = time.time()

    Tsa.append((end_time-start_time)/N)
    print('Total time taken by sa-rectangular L_{} MDP with mode {} is {},  relative cost {} \n'.format(p, 'EXACT', Tsa[-1], Tsa[-1]/nrt))
    
Tsa = np.array(Tsa)
# print('\n\n\n Summary of time and relative Time\n\n')
# print('non-robust time = {},  sa-rect time {} \n'.format(nrt, Tsa))
# print('Relative time of sa -rect w.r.t non robust = {}'.format(Tsa),'\n\n\n\n')


# ######   Excution time sa rectangular MDPs  ####
# Tlpsa = []

# for p in [1,'inf']:

#     start_time = time.time()
#     for n in range(N):
#         pg = RPG(p=p,rect='sa',mode='lp', tol=tol)
#     end_time = time.time()

#     Tlpsa.append((end_time-start_time)/N)
#     print('Total time taken by sa-rectangular L_{} MDP with mode {} is {},  relative cost {} \n'.format(p, 'LP', Tlpsa[-1], Tlpsa[-1]/nrt))
    
# Tsa = np.array(Tsa)
# print('\n\n\n Summary of time and relative Time\n\n')
# print('non-robust time = {},  sa-rect time {} \n'.format(nrt, Tsa))
# print('Relative time of sa -rect w.r.t non robust = {}'.format(Tsa),'\n\n\n\n')

######   Excution time s rectangular MDPs  ####
Ts = []

for p in [1,2,3,4,5,6,7,8,9,10,'inf']:
    
    start_time = time.time()
    for n in range(N):
        pg = RPG(p=p,rect='sa',tol=tol)
    end_time = time.time()

    Ts.append((end_time-start_time)/N)

    print('Total time taken by s-rectangular L_{} MDP  is {},  relative cost {} \n'.format(p, Ts[-1], Ts[-1]/nrt))
    

Ts = np.array(Ts)
print('\n\n\n Summary of time and relative Time\n\n')
# print('non-robust time = {},  s-rect time {}, s-rec \n'.format(nrt, Ts))

print('Relative time of sa -rect w.r.t non robust = {}'.format(Tsa/nrt),'\n\n\n\n')
print('Relative time of s -rect w.r.t non robust = {}'.format(Ts/nrt),'\n\n')

T = np.array([Tsa,Ts])/nrt 
np.savetxt('main_S{}A{}.txt'.format(pm.S,pm.A),T)

print('\n\n NEW EXPERIMENT \n\n')

# ######  Rate of convergence ######
# Error = [] # distance from optimal value function
# ROC = [] # rate of convergence
# for VI in [nr,sa1,sa2,sap,s1,s2,sp]:
#     epsilon = 1
#     v = pm.v0
#     V = []
#     for i in range(1000):
#         V.append(v)
#         v = VI(v,pm,p=4)
#     roc = np.sum(np.abs(V[100]-V[-1]))/np.sum(np.abs(V[0]-V[-1]))
#     roc = 1-np.exp(np.log(roc)/100)
#     ROC.append(roc)
# print(' Average rate of Convergence of first 100 iterates')
# print(ROC,'\n\n\n')
# # title = ['non-robust','sa1', 'sa2', 'sap','s1', 's2','sp']
# np.savetxt('time.txt',[Time, Time_rel,ROC])






