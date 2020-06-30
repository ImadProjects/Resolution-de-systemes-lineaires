
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import math as m
from matplotlib.pyplot import figure, show

## we arue using the cholesky method to solve Ax=b

def facto_dense(A, n):
    T = np.zeros(shape=(n,n))
    for i in range(n):
        for k in range(i+1):
            tmp_sum = sum(np.fromiter((T[i][j] * T[k][j] for j in range(k)),float))

            if (i == k):
                T[i][k] = np.sqrt(A[i][i] - tmp_sum)
            else:
                T[i][k] = (A[i][k] - tmp_sum)/(T[k][k])
    return T

## finding Y in the method

def descente(T,n,B):
    Y = np.zeros(shape=(n, 1))
    Y[0]= B[0]/T[0][0]
    for i in range(1,n):
        s = 0
        for j in range(i):
            s += T[i][j]*Y[j]
        Y[i] = (B[i] - s)/T[i][i]
    return Y

## finding X in the method

def remontee(T,n,Y):
    X = np.zeros(shape=(n, 1))
    X[n-1] = Y[n-1]/T[n-1][n-1]
    for i in range(n-2,-1,-1):
        s = 0
        for j in range(i+1,n):
            s += T[i][j] * X[j]
        X[i] = (Y[i] -s)/T[i][i]
    return X

#final resolution of Ax=b

def resolution_cholesky(A,B,n):
    T = facto_dense(-A,n**2)
    Y = descente(T,n**2,-B)
    X = remontee(T.T, n**2, Y)
    return X

# second method reolution with gradient 

def conjgrad(A, b,N):
    x=np.zeros(shape=(N**2,1))
    r=b-np.dot(A,x)
    p=r
    rsold=np.dot((np.conj(r).T),r)[0][0]

    for i in range(1, 10**6):
        Ap=np.dot(A,p)
        alpha=rsold/(np.dot(np.conj(p).T,Ap)[0][0])
        x=x+np.dot(alpha,p)
        r=r-np.dot(alpha,Ap)
        rsnew=np.dot((np.conj(r).T),r)[0][0]
        if np.sqrt(rsnew) < 10**(-10):
            break
        p=r+(rsnew/rsold)*p
        rsold=rsnew
    return x

# generate the derivative operator matrix

def generermatA(N):

    Matrix = np.zeros(shape=(N**2,N**2))
    for i in range(N**2):
        Matrix[i][i] = -4
        for j in range(N**2):
            if (j == i+1 and j%N != 0):
                Matrix[i][j] = 1
            if (i == j+1 and j%N != N-1):
                Matrix[i][j] = 1
            if (j == i+N):
                Matrix[i][j] = 1
            if (i == j+N):
                Matrix[i][j] = 1


    return Matrix

#generate f(x,y) <=> B

def genererB(f,N):
    B = np.zeros(shape=(N**2, 1))
    for i in range(N):
        for j in range(N):
            B[i*N+j] = f(i,j,N)
    return B
#function for center radiator

def radiateur_centre(i,j,N):
    if (i > N/2-5 and i < N/2+5 and j > N/2-5 and j < N/2+5):
        return -25
    return 0
#function tjat generate a north wall

def mur_nord(i,j,N):
    if (i > N-2):
        return -25
    return 0
#generate solotion according to the methode chosen

def gen_x(N, fonction, methode):
    A = generermatA(N)
    B = genererB(fonction,N)
    if (methode == "numpy"):
        x = np.linalg.solve(A,B)
    if (methode == "gradient"):
        x = conjgrad(A,B,N)
    if (methode == "cholesky"):
        x = resolution_cholesky(A,B,N)
    return x

#tranform a vector into a matrix

def vect_to_mat(x,N):
    T = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(N):
            T[i][j] = x[i*N+j][0]
    return T


def Eqchaleur(fonction, N, methode):
    return vect_to_mat(gen_x(N, fonction, methode), N)


def afficher(fonction, N, p, methode):
    fig=figure()
    pc = plt.pcolormesh(Eqchaleur(fonction, N, methode), cmap='hot' , shading='bilinear')
    fig.colorbar(pc)
    plt.xticks([x for x in range(0,N-1,int(N/10))]+[N-1], [i/10 for i in range(10)]+[1.0])
    plt.yticks([x for x in range(0,N-1,int(N/10))]+[N-1], [i/10 for i in range(10)]+[1.0])

    plt.show()


#afficher(mur_nord, 80, 0, "gradient" )
#afficher(radiateur_centre,80,1, "gradient")
#afficher(radiateur_centre, 40, 1, "cholesky" )

def calculer_ec_rel(fonction, N, methode):

    norme_numpy = np.linalg.norm(Eqchaleur(fonction, N, "numpy"))
    norme_fonction = np.linalg.norm(Eqchaleur(fonction, N, methode))

    ecart = abs(norme_fonction - norme_numpy)/norme_numpy

    return ecart


