import numpy as np
import math as m
import unittest
import matplotlib.pyplot as plt
import random

###******************/PART II : Conjugate gradient method

"""QUESTION 3"""
"""Resolution of the linear system Ax=b by the conjugate gradient method where A is a symetric positive definite matrix and b a vector (without preconditioning)"""

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
    return x,tab_x,iter             #x is the solution
                                    #tab_x is a list regrouping the values of x at each iteration
                                    #iter is a list regrouping the iterations before arriving to the final solution. 


"""QUESTION 4"""
"""Resolution of the linear system with preconditioning. Two preconditionners were implemented: using the Jacobi method and the incomplete cholesky method"""

##******************* Preconditionning with the ---Incomplete Cholesky Method-

def somme(T, i, j):
    return sum(np.fromiter((T[i][k] * T[j][k] for k in range(j)),float))

def facto_dense_inc(A):                   #this function is the implementation of the incomplete cholesky method seen in the Part I  
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

def preconditioner(A):                    #this function returns a matrix preconditionned with the incomplete cholesky method
    T= facto_dense_inc(A)
    return np.dot(T,np.transpose(T))


##******************* Preconditionning with the ---Jacobi Method-
    
def generate_jacobi_matrix(A):                       #M is a diagonal matrix whose coefficients are the inverse of the diagonal 
                                                     #coefficients of the matrix A
    M=np.zeros((len(A),len(A)))
    for i in range(len(A)):
        if A[i][i]!=0:
            M[i][i]=1/A[i][i]
    return M


def PreconditionedConjgrad(A,b,x):
    tab_x=[x]
    iter=[0]
    r=b-np.dot(A,x)
    M=generate_jacobi_matrix(A)          # we choose the jacobi method but we can use the incomplete cholesky method 
                                         # by puting M=np.linalg.inv(preconditionner(A)
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
    return x,tab_x,iter             #x is the solution
                                    #tab_x is a list regrouping the values of x at each iteration
                                    #iter is a list regrouping the iterations before arriving to the final solution. 
    
    
def MdpGenerator(size) :          #this auxiliary function generates randomly, for a given size, a matrix A
                                  #symetric definite positive 
                                  #T is an upper triangular matrix with positive diagonal coefficients 
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

##Test 1: 

class Test_gradient(unittest.TestCase):
    def test_conjgrad(self):                            # we test conjgrad with a matrix whose size is 2*2
        A= np.array([[4,1],[1,3]])
        b= np.array([1,2])
        x= np.array([2,1])                              # initialisation of x
        expected= np.array([0.0909,0.6363])             # expected is the exact solution 
        result= conjgrad(A,b,x)                         # result is the solution given by conjgrad
        for i in range(len(A)):                         # we compare result and expected
                self.assertAlmostEqual(result[0][i],expected[i],3)
    def test_conjgrad_precond(self):                    # we test preconditionedconjgrad with a matrix whose size is 2*2
        A= np.array([[4,1],[1,3]])
        b= np.array([1,2])
        x= np.array([2,1])
        expected= np.array([0.0909,0.6363])
        result= PreconditionedConjgrad(A,b,x)
        print()
        for i in range(len(A)):                       #comparison between the solution with preconditionning and the exact solution             
            self.assertAlmostEqual(result[0][i],expected[i],3)    


##Test 2: Conjugate gradient VS Preconditionned Conjugate gradient

##We generate a curve representing the variations of the absolute error according to the number of iterations
##absolute error = the difference in magnitudes between the expected solution and the solution found by the algorithm 

def test_conjgrad(size):
    R1=[]
    R2=[]
    A=MdpGenerator(size)
    Xs = np.random.rand(size,1)
    x1=np.zeros((size,1))
    x2=np.zeros((size,1))
    b=np.dot(A,Xs)
    s1=conjgrad(A,b,x1)
    s2=PreconditionedConjgrad(A,b,x2)
    for i in range(len(s1[1])):
        R1+=[abs(np.linalg.norm(s1[1][i])-np.linalg.norm(Xs))]
        R2+=[abs(np.linalg.norm(s2[1][i])-np.linalg.norm(Xs))]
    return R1,R2,s1[2]

def show_tests(size):
    t=test_conjgrad(size)
    plt.plot(t[2],t[1], label="PreConjgrad_Method")
    plt.plot(t[2],t[0], label="Conjgrad_Method")
    plt.xlabel('Iterations')
    plt.ylabel('Absolute Error')
    plt.legend()
    plt.show()
    
#show_tests(20)                              #we run the test with a matrix A whose size is 20*20

##Test 3: Conjugate gradient VS Preconditionned Conjugate gradient VS linalg.solve

def test_conjgrad_linalg(size):
                                    #In this test we generate the solutions given by np.linalg, conjgrad and    
                                    #preconditionnedconjgrad for different size of matrix
                                    #we generate a graph with 3 curves which show how the absolute error varies for each method
    R1=[]
    R2=[]
    R3=[]
    iter=[]
    for i in range(25,size):                                          #we iterate on the matrix size 
        iter+=[i]
        A=MdpGenerator(i)
        Xs = np.random.rand(i,1)                                      #Xs is the exact solution
        b=np.dot(A,Xs)
        
        xConjgrad=np.zeros((i,1))
        xConjgrad=conjgrad(A,b,xConjgrad)                             #we save the solution, for the system, given by conjgrad 
        
        xPreconjgrad=np.zeros((i,1))
        xPreconjgrad=PreconditionedConjgrad(A,b,xPreconjgrad)         #we save the solution given by Preconditionnalconjgrad     
                                                                    
        xLinalg=np.linalg.solve(A,b)                                  #we save the solution given by linalg.solve
        
        R1+=[abs(np.linalg.norm(xConjgrad[0])-np.linalg.norm(Xs))]    #we stock the absolute error for each solution x 
        R2+=[abs(np.linalg.norm(xPreconjgrad[0])-np.linalg.norm(Xs))]
        R3+=[abs(np.linalg.norm(xLinalg)-np.linalg.norm(Xs))]
    plt.plot(iter, R1, label="Conjgrad Method")
    plt.plot(iter, R3, label="Linalg from numpy")
    plt.plot(iter, R2, label="PreConjgrad Method")
    plt.xlabel('Size of matrix')
    plt.ylabel('Absolute error')
    plt.title('Absolute error in the resolution of Ax=b') 
    plt.legend()
    plt.show()
    
test_conjgrad_linalg(50)                                #we run the test for matrix whose size varies from 25*25 to 49*49

# unittests
if __name__ == '__main__':
    unittest.main(Test_gradient(), verbosity = 2)
