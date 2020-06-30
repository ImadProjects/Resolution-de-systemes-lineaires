import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

#PARTIE 3

def ConjGrad(A,b,x):
    if (np.linalg.norm(b) == 0):
        return 0
    r = b - np.dot(A,x); # r est un vecteur
    p = r
    rsoldmat = np.dot(r.transpose(), r); # rsoldmat est un scalaire
    rsold = rsoldmat

    for i in range(1000000):
        Ap = np.dot(A,p); # Ap est un vecteur
        pAp = np.dot(p.transpose(), Ap); # pAp est un scalaire (delta pour moi)
        alpha = rsold/(pAp) # alpha est un scalaire
        x += alpha*p;
        r -= alpha*Ap
        rsnew = np.dot(r.transpose(), r); # produit scalaire
        if (np.sqrt(rsnew)< 1e-10):
            break;
        p = r + rsnew/rsold*p
        rsold = rsnew

    return x

def generate_matA(N): #retourne la matrice A
    A = np.zeros((N*N,N*N))
    for i in range(N*N):
        A[i][i]= -4.0;
        if(i+1 < N*N and i%N != N-1):
            A[i+1][i] = 1.0
        if(i-1 >= 0 and i%N != 0):
            A[i-1][i] = 1.0
        if(i+N < N*N  ):
            A[i+N][i] = 1.0
        if(i-N >= 0 ):
            A[i-N][i] = 1.0
    return A


def generate_vect_B_Rmilieu(N):#retourne le vecteur B pour Radiateur milieu (N/2,N/2)
    b = np.zeros((N*N,1))
    b[N*N/2 + N/2] = 1.0 #cas du radiateur au milieu(i=N/2,j=N/2)
    return b

def generate_vect_B_Rnord(N):   #retourne le vecteur B pour radia au nord (i=0)
    b = np.zeros((N*N,1))
    #radiateur en haut (nord)
    for j in range(N):
            b[N*(N-1) +j] = 1.0 #sources de chaleur se situent dans la ligne en haut
    return b

def resolution_sys_cha_milieu(N):#resolution systeme radiateur placé au mileu
    b = generate_vect_B_Rmilieu(N)#
    A = generate_matA(N)#
    x = np.zeros((N*N,1))
    h = (1/float(N+1))
    A =(h**2)*A
    ConjGrad(A,b,x)
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            M[i][j] = -x[i*N+j]
    return M
      
    

def resolution_sys_cha_nord(N):#resolution systeme mur chaud placé au nord 
    b = generate_vect_B_Rnord(N)
    A = generate_matA(N)
    x = np.zeros((N*N,1))
    h = float(1/float(N+1))
    A = (h**2)*A 
    
    ConjGrad(A,b,x)
    M = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            M[i][j] = -x[i*N+j]
    return M

#affichage des resultats
"""on prend par exemple N=10 pour l'affichage des resultats des deux problemes """
N = 10
#m = resolution_sys_cha_milieu(N)
m=resolution_sys_cha_nord(N)
picture = plt.imshow(m)
axes = plt.gca()
axes.xaxis.set_ticklabels(['x=0','x=0.1','x=0.2','x=0.3','x=0.4','x=0.5', 'x=0.6','x=0.7','x=0.8','x=0.9','x=1'])
axes.yaxis.set_ticklabels(['y=0','y=0.1', 'y=0.2','y=0.3','y=0.4','y=0.5','y=0.6','y=0.7','y=0.8','y=0.9','y=1'])
axes.set_xlim(0,9)

axes.set_ylim(0,9)
plt.show(picture)
#A = diag(4.0*ones(N*N)) - diag(ones(N*N-1), -1) - diag(ones(N*N-1), 1) - diag(ones(N*N-N), -N) - diag(ones(N*N-N), N)
