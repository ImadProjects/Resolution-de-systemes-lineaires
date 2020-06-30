import numpy as np
import math as m
import unittest
import matplotlib.pyplot as plt
import random

###******************/PART II : Conjugate gradient method

"""QUESTION 3"""
"""Resolution of the linear system Ax=b by the conjugate gradient method where A is a symmetric positive definite matrix and b a vector (without preconditioning)"""

def conjgrad(A,b,x) :
    r=b-np.dot(A,x)
    p=r
    rsOld= float(np.dot(np.transpose(r),r))
    tab_x=[x]
    iter=[0]
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



"""QUESTION 4"""
"""Resolution of the linear system Ax=b by the conjugate gradient method where A is a symmetric positive definite matrix and b a vector (with preconditioning)"""

#we can generate a precondition by this method
def generate_jacob_matrix(A):
    M=np.zeros((len(A),len(A)))
    for i in range(len(A)):
        if A[i][i]!=0:
            M[i][i]=1/A[i][i]
    return M

def somme(T, i, j):
    return sum(np.fromiter((T[i][k] * T[j][k] for k in range(j)),float))

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

# generating a precondition with incomplete cholsky method
def preconditioner(A):
    T= facto_dense_inc(A)
    return np.dot(T,np.transpose(T))

def PreconditionedConjgrad(A,b,x):
    tab_x=[x]
    iter=[0]
    r=b-np.dot(A,x)
    M=generate_jacob_matrix(A) #lwe can use either the incomplete cholsky precondition or the jacob matrix precondition
    #or M=np.linalg.inv(preconditionner(A))
    z=np.dot(M,r)
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
        z2=np.dot(M,r2)  
        beta=(np.dot(np.transpose(z2),r2))/(np.dot(np.transpose(z),r))
        p=z2+beta*p
        z=z2
        r=r2
    return x,tab_x,iter

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

##****************TEST: Conjugate gradient and Preconditionned Conjugate gradient methods 


class Test_gradient(unittest.TestCase):
    def test_conjgrad(self):
        A= np.array([[4,1],[1,3]])
        b= np.array([1,2])
        x= np.array([2,1])
        expected= np.array([0.0909,0.6363])
        result= conjgrad(A,b,x)
        for i in range(len(A)):
                self.assertAlmostEqual(result[0][i],expected[i],3)
    def test_conjgrad_precond(self):
        A= np.array([[4,1],[1,3]])
        b= np.array([1,2])
        x= np.array([2,1])
        expected= np.array([0.0909,0.6363])
        result= PreconditionedConjgrad(A,b,x)
        print()
        for i in range(len(A)):
            self.assertAlmostEqual(result[0][i],expected[i],3)    

##We generate a curve representing the variations of the relative error according to the number of iterations
##relative error = the difference in magnitudes between the expected solution and the solution found by the algorithm 

def test1_conjgrad(size):
    R=[]
    A=MdpGenerator(size)
    Xs = np.random.rand(size,1)
    x=np.zeros((size,1))
    b=np.dot(A,Xs)
    s=conjgrad(A,b,x)
    for i in range(len(s[1])):
        R+=[abs(np.linalg.norm(s[1][i])-np.linalg.norm(Xs))]
    return R,s[2]

def test2_conjgrad(size):
    R=[]
    A=MdpGenerator(size)
    Xs = np.random.rand(size,1)
    x=np.zeros((size,1))
    b=np.dot(A,Xs)
    s=PreconditionedConjgrad(A,b,x)
    for i in range(len(s[1])):
        R+=[abs(np.linalg.norm(s[1][i])-np.linalg.norm(Xs))]
    return R,s[2]


def show_tests(size):
    s=test1_conjgrad(size)
    t=test2_conjgrad(size)
    plt.plot(t[1],t[0], label="PrecCond_Conjgrad_Method")
    plt.plot(s[1],s[0], label="Conjgrad_Method")
    plt.xlabel('nbrs iterations')
    plt.ylabel('Relative Error')
    plt.legend()
    plt.show()
    
#show_tests(15)

#In this test we generate the solutions given by np.linalg, conjgrad and precond_conjgrad 
#for matrix A whose size is varying from 25 to 'size-1'
##relative error = the difference in magnitudes between the expected solution and
##the solution found by the algorithm

def TestConjgrad(size):
    R1=[]
    R2=[]
    R3=[]
    iter=[]
    for i in range(25,size):
        iter+=[i]
        A=MdpGenerator(i)
        Xs = np.random.rand(i,1)
        x_conjgrad=np.zeros((i,1))
        b=np.dot(A,Xs)
        x_conjgrad=conjgrad(A,b,x_conjgrad)
        conjgrad(A,b,x_conjgrad)
        x_resultpreconjgrad=PreconditionedConjgrad(A,b,x_conjgrad)
        PreconditionedConjgrad(A,b,x_conjgrad)
        x_linalg=np.linalg.solve(A,b)
        R1+=[abs(np.linalg.norm(x_resultconjgrad[0])-np.linalg.norm(Xs))]
        R2+=[abs(np.linalg.norm(x_resultpreconjgrad[0])-np.linalg.norm(Xs))]
        R3+=[abs(np.linalg.norm(x_linalg)-np.linalg.norm(Xs))]
    plt.plot(iter, R1, label="Conjgrad Method")
    plt.plot(iter, R3, label="Linalg from numpy")
    plt.plot(iter, R2, label="PreConjgrad Method")
    plt.xlabel('Size of matrix')
    plt.ylabel('Relative error')
    plt.title('Relative error in the resolution of Ax=b') 
    plt.legend()
    plt.show()
    plt.savefig('relative_error.png')
    
TestConjgrad(50)

# unittests
if __name__ == '__main__':
    unittest.main(Test_gradient(), verbosity = 2)