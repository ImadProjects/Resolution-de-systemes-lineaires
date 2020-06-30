#!/usr/bin/python3.4
import numpy as np
import math as m
import random as rand
import unittest

# -----------------------Question 1---------------------------

def somme(T, i, j):
	return sum(np.fromiter((T[i][k] * T[j][k] for k in range(j)),float))

def dense_factorization(A):
    (n, n1) = A.shape
    T = np.zeros(shape=(n,n))
    for i in range(n):
        for k in range(i+1):
            tmp_sum = somme(T, i, k)

            if (i == k):
                T[i][k] = np.sqrt(A[i][i] - tmp_sum)
            else:
                T[i][k] = (A[i][k] - tmp_sum)/(T[k][k])
    return T


# -----------------------Question 3---------------------------

def is_pos_mat(M):
    liste = np.linalg.eigvals(M)
    for i in range(len(liste)):
        if liste[i]<0:
            return 0
    return 1        

def count_non_zero(M):
    (n1, n2) = M.shape
    count = 0
    for i in range(n1):
        for j in range(n2):
            if i != j and M.item(i, j) != 0:
                count = count + 1
    return count

# n :Size of the matrix , n_termes : number of extra-diagonals terms non-zeros
def generate_matrix(n, n_termes):
	if n_termes > n * (n - 1) or n_termes % 2 != 0:
		raise Exception("Invalid n_termes value")
	
	res = np.diag(np.random.uniform(0.1, 1, n))
	i = 0
	diag = 1 # number of the diagonal 
	e = 0 # Index in the diagonal
	while i < n_termes // 2:
		res.itemset(e, e + diag, rand.uniform(0.1, 10)) #add a number
		i = i + 1
		e = e + 1
		if e >= n - diag: # If it is the end of the diagonal , skip to the next
			e = 0
			diag = diag + 1
			
	return np.matmul(res, res.transpose())


# -----------------------Question 4---------------------------
#TODO ecrire somme
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


# -----------------------Question 5---------------------------


# apparently, the t(T-1)*T-1 is not a good cond for matrix ; but needs more testing

#the fact that COND for the newer version = COND for the older version is logical since they represent the same matrix


# -----------------------Tests---------------------------

class Test_Cholesky(unittest.TestCase):
    def test_facto(self):
        A = np.array([[1,1,1,1],
                      [1,2,2,2],
                      [1,2,3,3],
                      [1,2,3,4]])
        L = dense_factorization(A)
        self.assertTrue(np.allclose(A, np.matmul(L, L.T)))
        
    def test_generatemat(self):
        for i in range(0, 20, 2):
            M = generate_matrix(100, i)
            self.assertTrue(is_pos_mat(M))
            self.assertTrue(np.allclose(M, M.T)) # assert M is symetric
            self.assertEqual(count_non_zero(M), i)

    def test_facto_inc(self):
        A1 = np.array([[1,1,1,1],
                      [1,2,2,2],
                      [1,2,3,3],
                      [1,2,3,4]])
        L1 = facto_dense_inc(A1)
        self.assertTrue(np.allclose(A1, np.matmul(L1, L1.T)))
        
        A2 = generate_matrix(10,4)
        L2 = facto_dense_inc(A2)
        self.assertTrue(np.allclose(A2, np.matmul(L2, L2.T)))

    def test_cond(self):
        M=generate_matrix(4,4)
        MAT_NEW = facto_dense_inc(M)
        MAT_OLD = dense_factorization(M)
        print ("-------------------------------------------------------------" )
        print ("The factorial matrix T by the incomplete method :")
        print (MAT_NEW)
        print ("-------------------------------------------------------------" )
        print ("The factorial matrix T by the first method :")
        print (MAT_OLD)
        print ("COND FOR NEWER VERSION :")
        COND_NEW = np.matmul(np.linalg.inv(MAT_NEW).T, np.linalg.inv(MAT_NEW))
        print (np.linalg.cond(np.matmul(COND_NEW, M)))
        print ("COND FOR OLDER VERSION :")
        COND_OLD = np.matmul(np.linalg.inv(MAT_OLD).T, np.linalg.inv(MAT_OLD))
        print (np.linalg.cond(np.matmul(COND_OLD, M)))
        print("MEANWHILE COND MATRICE :")
        print (np.linalg.cond(M))
        self.assertTrue(np.linalg.cond(np.matmul(COND_NEW, M)) < np.linalg.cond(M))
        self.assertTrue(np.linalg.cond(np.matmul(COND_OLD, M)) < np.linalg.cond(M))
        

if __name__ == '__main__':
    unittest.main(Test_Cholesky(), verbosity = 2)
