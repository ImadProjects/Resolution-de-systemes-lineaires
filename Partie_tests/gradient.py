# -*-coding:latin-1 -*   j
import numpy as np
import scipy as sc
import math as m
import time
import matplotlib.pyplot as plt

def conjgrad(A,b,x) :
    r=b-np.dot(A,x)
    p=r
    rsOld= float(np.dot(np.transpose(r),r))
    for i in range(1,100001):
        Ap=np.dot(A,p)
        n = float(np.dot(np.transpose(p),Ap))
        alpha=rsOld/n
        x=x+ alpha*p
        r=r- alpha*Ap
        rsNew=float(np.dot(np.transpose(r),r))

        if m.sqrt(rsNew) < 1e-10 :
            break
        p=r+rsNew/rsOld*p
        rsOld=rsNew
    return x

A=np.array([[4,1],[1,3]])
b=np.array([[1],[2]])
x=np.array([[0],[0]])
init_time1 = time.time()
x=conjgrad(A,b,x)
final_time1= time.time()
print(x[0][0])
print(x[1][0])


def preconditioned_conjgrad(A,b,x):
    r=b-np.dot(A,x)
    z=np.dot(np.linalg.inv(A),r)
    p=z
    for k in range(1,100001):
        alpha=(np.dot(np.transpose(r),z))/(np.dot(np.transpose(p),np.dot(A,p)))
        x=x+ alpha*p
        r2=r-alpha*(np.dot(A,p))
        rsNew=float(np.dot(np.transpose(r2),r2))

        if m.sqrt(rsNew) < 1e-10 :
            break
        z2=np.dot(np.linalg.inv(A),r2)  
        b=(np.dot(np.transpose(z2),r2))/(np.dot(np.transpose(z),r))
        z=z2
        r=r2
    return x        



A2=np.array([[4,1],[1,3]])
b2=np.array([[1],[2]])
x2=np.array([[0],[0]])
init_time2 = time.time()
x2=preconditioned_conjgrad(A2,b2,x2)
final_time2= time.time()
print(x2[0][0])
print(x2[1][0])


def test_convergence_rapidity(A,b,x):
    print ("temps execution conjgrad =", final_time1-init_time1)
    print ("temps execution preconditioned_conjgrad = ", final_time2-init_time2)

test_convergence_rapidity(A2,b2,x2)

from scipy import linalg
from random import randint




def mdp_generator(i,j):
    B = np.random.rand(i,j)
    A = np.dot(B,B.transpose())
    return A

def conjgrad_tst(size):
    resFalse = 0
    iter = 0
    tab_err=[]
    for i in range(2,size+1):
        for j in range(2,size+1):
            iter+=1
            A = mdp_generator(i,j)
            Xs = np.random.rand(i,1)
            x=np.zeros((i,1))
            b=np.dot(A,Xs)
            #On compare la valeur absolue de la différence des norme entre la solution fournie par conjgrad et Xs qu'on connait déjà
            #à une tolérance d'erreur de 10**(-10)
            err=abs(np.linalg.norm(conjgrad(A,b,x))-np.linalg.norm(Xs))
            tab_err.append(err)
            if (err > 10**(-10)):
                resFalse+=1
    print("Nombres de systèmes résolus sans préconditionneur : %d/%d" %(iter-resFalse,iter))
    #return resFalse pr generer le graph du nombre de faux en fonction de la taille
    #return tab_err
    return iter
 

print("la taille est ",conjgrad_tst(6))
   
def graph_generator_norm():
    X=np.arange(0,(5-1)**2,1)  #matrice de taille 5*5
    Y=conjgrad_tst(6)
    plt.xlabel('Nombre d itérations')
    plt.ylabel("Erreur entre la solution correct et le résultat de l algorithme")
    plt.title("Variation de la différence entre le résultat attendu et le résultat calculé par l algorithme au fil des itérations")
    plt.plot(X,Y) 
    plt.show()
    plt.savefig('erreur_norme.png') 
    
def graph_generator_size():
    X=np.arange(0,(5-1)**2,1)  #matrice de taille 5*5
    Y=np.array([conjgrad_tst(i) for i in X])
    plt.xlabel('Taille de la matrice A')
    plt.ylabel("Nombre de résultats faux")
    plt.title("Représentation du nombre de résultats faux du système Ax=b en fonction de la taille de la matrice A")
    plt.plot(X,Y) 
    plt.show()
    plt.savefig('faux_size.png')        


#graph_generator_norm()

#graph_generator_size()


