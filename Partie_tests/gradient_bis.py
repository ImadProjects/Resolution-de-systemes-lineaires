import numpy as np
import math as m
from timeit import timeit

import matplotlib.pyplot as plt

def conjgrad(A,b,x) :
    r=b-np.dot(A,x)
    p=r
    rsOld= float(np.dot(np.transpose(r),r))
    tab_x=[]
    iter=[]
    for i in range(1,100001):
        Ap=np.dot(A,p)
        n = float(np.dot(np.transpose(p),Ap))
        alpha=rsOld/n
        x=x+ alpha*p
        r=r- alpha*Ap
        rsNew=float(np.dot(np.transpose(r),r))
        tab_x+=[x]
        iter+=[i]
        if m.sqrt(rsNew) < 1e-10 :
            break
        p=r+rsNew/rsOld*p
        rsOld=rsNew
    return x,tab_x,iter 


def preconditioned_conjgrad(A,b,x):
    r=b-np.dot(A,x)
    iter=[]
    tab_x=[]
    z=np.dot(np.linalg.inv(A),r)
    p=z
    for k in range(1,100001):
        alpha=(np.dot(np.transpose(r),z))/(np.dot(np.transpose(p),np.dot(A,p)))
        x=x+ alpha*p
        tab_x+=[x]
        iter+=[k]
        r2=r-alpha*(np.dot(A,p))
        rsNew=float(np.dot(np.transpose(r2),r2))
        if m.sqrt(rsNew) < 1e-10 :
            break
        z2=np.dot(np.linalg.inv(A),r2)  
        b=(np.dot(np.transpose(z2),r2))/(np.dot(np.transpose(z),r))
        z=z2
        r=r2
    return x,tab_x,iter


def gen_MSDP(size) : 
    B = np.random.rand(size,size)
    A = np.dot(B,B.transpose())
    return A
 
def test1_conjgrad(size):
    R=[]
    A=gen_MSDP(size)
    Xs = np.random.rand(size,1)
    x=np.zeros((size,1))
    b=np.dot(A,Xs)
    s=conjgrad(A,b,x)
    for i in range(len(s[1])):
        R+=[abs(np.linalg.norm(s[1][i])-np.linalg.norm(Xs))]
    return R,s[2]

def test2_conjgrad(size):
    R=[]
    A=gen_MSDP(size)
    Xs = np.random.rand(size,1)
    x=np.zeros((size,1))
    b=np.dot(A,Xs)
    s=preconditioned_conjgrad(A,b,x)
    for i in range(len(s[1])):
        R+=[abs(np.linalg.norm(s[1][i])-np.linalg.norm(Xs))]
    return R,s[2]


#s=test1_conjgrad(100)
t=test2_conjgrad(10)


#plt.plot(s[1],s[0])
#plt.xlabel('nbrs iterations')
#plt.ylabel('Erreur relative')
#plt.show() 

plt.plot(t[1],t[0])
plt.xlabel('nbrs iterations(precondit)')
plt.ylabel('Erreur relative(precondit)')
plt.show()


A=np.array([[4,1],[1,3]])
b=np.array([[1],[2]])
x=np.array([[0],[0]])
s=conjgrad(A,b,x)
print(s[0][0][0])
print(s[0][1][0])


A2=np.array([[4,1],[1,3]])
b2=np.array([[1],[2]])
x2=np.array([[0],[0]])
init_time2 = time.time()
x2=preconditioned_conjgrad(A2,b2,x2)
final_time2= time.time()
print(x2[0][0])
print(x2[1][0])
