import numpy as np
import math as m
import time
import matplotlib.pyplot as plt
import unittest

"""QUESTION 3"""
"""Resolution of the linear system Ax=b by the conjugate gradient method where A is a symmetric positive definite matrix and b a vector (without preconditioning)"""


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
            max_iteration=i
            break
        p=r+rsNew/rsOld*p
        rsOld=rsNew
    #print('number of iteration needed to find X',max_iteration)    
    return x#x is the solution
                        #tab_x is a list regrouping the values of x at each iteration
                        #iter is a list regrouping the iterations before arriving at the final solution. 
"""QUESTION 4"""

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
            max_iteration=k
            break
        z2=np.dot(np.linalg.inv(M),r2)  
        b=(np.dot(np.transpose(z2),r2))/(np.dot(np.transpose(z),r))
        z=z2
        r=r2
    print('number of iteration needed to find X',max_iteration)    
    return x

##****************TEST: Conjugate gradient and Preconditionned Conjugate gradient methods 
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
'''
def measure_execution_time():
        A= np.array([[4,1],[1,3]])
        b= np.array([1,2])
        x= np.array([2,1])
        init_time1= time.time()
        PreconditionedConjgrad(A,b,x)
        final_time1= time.time()   
        init_time2= time.time()
        conjgrad(A,b,x)
        final_time2= time.time()     
        print('time needed for non precondionned method to find X',final_time1-init_time1)        
        print('time needed for precondionned method to find X',final_time2-init_time2)    
measure_execution_time()'''
#s1=TestConjgrad_1(100)
##We generate a curve representing the variations of the relative error according to the number of iterations
##relative error = the difference in magnitudes between the expected solution and the solution found by the algorithm 

#------------->inutile de tracer les tests suffisent (mais a vous de voir)

#plt.plot(s1[1],s1[0])
#plt.xlabel('Number of iterations')
#plt.ylabel('Relative error')
#plt.title('Variations of the relative error in the resolution of Ax=b') 
#plt.show() 



if __name__ == '__main__':
    unittest.main(Test_gradient(), verbosity = 2)
