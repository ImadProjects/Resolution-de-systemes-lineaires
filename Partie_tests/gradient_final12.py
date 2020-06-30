import numpy as np
import math as m
import time
import matplotlib.pyplot as plt
import unittest
import random

###*************************************
###*************************************
###*************************************

###******************/PART II : Conjugate gradient method

"""QUESTION 3"""
"""Resolution of the linear system Ax=b by the conjugate gradient method where A is a symmetric positive definite matrix and b a vector (without preconditioning)"""


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
    return x                      #x is the solution found by Conjugate gradient Method
                        

def MdpGenerator(size) :          #this auxiliary function generates, for a given size, a matrix A
                                  #symmetric definite positive 
                                  #T is an upper trinagular matrix with positive diagonal coefficients 
                                  #The diagonal coefficients of A are generated randomly between 5 and 10      
    T=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            if j>=i:
                T[i,j] = random.randint(5,10)
            else:
                T[i,j] = 0
    A = np.dot(T,T.transpose())
    return A

"""QUESTION 4"""
"""Resolution of the linear system Ax=b by the conjugate gradient method where A is a symmetric positive definite matrix and b a vector (with preconditioning)"""

def somme(T, i, j):
    return sum(T[i][k] * T[j][k] for k in range(j))

def facto_dense_inc(A):
    (n, n1) = A.shape
    T = np.zeros((n,n))
    for i in range(n):
        for j in range(i + 1):
            if A[i][j] != 0:
                if i==j:
                    T[i][j] = m.sqrt((A[i][i] - somme(T, i, j)))
                else:
                    T[i][j] = (A[i][j] - somme(T, i, j)) / T[j][j]
    return T


def preconditioner(A):
    T= facto_dense_inc(A)
    return np.dot(T,np.transpose(T))

def PreconditionedConjgrad(A,b,x):
    r=b-np.dot(A,x)
    M=preconditioner(A)
    z=np.dot(np.linalg.inv(M),r)
    p=z
    for k in range(1,100001):
        alpha=(np.dot(np.transpose(r),z))/(np.dot(np.transpose(p),np.dot(A,p)))
        x=x+ alpha*p
        r2=r-alpha*(np.dot(A,p))
        rsNew=float(np.dot(np.transpose(r2),r2))
        if m.sqrt(rsNew) < 1e-10 :
            break
        z2=np.dot(np.linalg.inv(M),r2)  
        b=(np.dot(np.transpose(z2),r2))/(np.dot(np.transpose(z),r))
        z=z2
        r=r2
    return x


##****************TEST: Conjugate gradient method 

class Test_gradient(unittest.TestCase):
    def test_conjgrad(self):
        A= np.array([[4,1],[1,3]])
        b= np.array([1,2])
        x= np.array([2,1])
        expected= np.array([0.0909,0.6363])
        result= conjgrad(A,b,x)
        for i in range(len(A)):
                self.assertAlmostEqual(result[i],expected[i],3)
    
    def test_conjgrad_precond(self):
        A= np.array([[4,1],[1,3]])
        b= np.array([1,2])
        x= np.array([2,1])
        expected= np.array([0.0909,0.6363])
        result= PreconditionedConjgrad(A,b,x)
        print()
        for i in range(len(A)):
            self.assertAlmostEqual(result[i],expected[i],3)  
                

def test_conjgrad_linalg(size):     #In this test we compare the relative error generated by np.linalg and conjgrad()
                                    #The size of the matrix A varies from 25 to 'size-1'  
                                    ##relative error = the difference in magnitudes between the expected solution and         
                                    ##the solution found by the algorithm  
    R1=[] 
    R2=[]
    iter=[]
    for i in range(25,size):
        iter+=[i]
        A=MdpGenerator(i)
        Xs = np.random.rand(i,1)
        x_conjgrad=np.zeros((i,1))
        b=np.dot(A,Xs)
        x_conjgrad=conjgrad(A,b,x_conjgrad)
        x_linalg=np.linalg.solve(A,b)
        R1+=[abs(np.linalg.norm(x_conjgrad)-np.linalg.norm(Xs))]
        R2+=[abs(np.linalg.norm(x_linalg)-np.linalg.norm(Xs))]
    plt.plot(iter, R1, label="Conjgrad Method")
    plt.plot(iter, R2, label="Linalg from numpy")
    plt.xlabel('Size of matrix')
    plt.ylabel('Relative error')
    plt.title('Relative error in the resolution of Ax=b') 
    plt.legend()
    plt.show()
    #plt.savefig('relative_error.png')
    
test_conjgrad_linalg(50)                              #we run the test for size=50

###*************************************
###*************************************
###*************************************

if __name__ == '__main__':
    unittest.main(Test_gradient(), verbosity = 2)
