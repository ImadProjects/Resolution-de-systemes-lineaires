import numpy as np
from math import *
import matplotlib.pyplot as plt

L=[np.log(1+1/pow(10,i)) for i in range (6)]; # On initialise le tableau L
A=[atan(1/pow(10,i)) for i in range (5)]

#-------------------------Ln(x)--------------------------------

def ln1(x):
    if x<=0 :
        print(x, "n'appartient pas au domaine de définition de ln ") 
        exit(0)
    n=0
    
    #Mettre x dans l'intervalle [1,10[
    while x<1 or x>=10:
        if x<1:
            x=x*10
            n-=1
        else:
             x=x/10
             n+=1
    # x est dans l'intervalle [1,10[ et ln(x)= ln(x)+n*ln(10)
    
    # On calcule ln(x)
    k=0;y=0;p=1
    while k < 6:
        while (x >= p+p/pow(10,k)):
            y = y + L[k]
            p = p + p/pow(10,k)
        k+=1
    return y+(x/p-1) + n*np.log(10)

def test_ln():    
    a=0.0000000000001
    b=20
    y=np.linspace(a,b,101)
    z=np.linspace(a,b,101)
    for i in range (0,101):
        z[i]=abs(log(y[i])-ln1(y[i]))*100/abs(log(y[i]))
    plt.subplot(221)
    plt.plot(y,z)
    plt.title("Erreur relative à la fonction ln")
    plt.xlabel("y")
    plt.ylabel("Erreur relative ")

#-------------------------Exp(x)--------------------------------    
def exp1(x):
    n=0
    while x<0 or x>=np.log(10):
        if x<0:
            x = x+np.log(10)
            n -= 1
        else:
            x = x-np.log(10)
            n += 1
    # x est bien dans l'intervalle [0,ln(10)]
    k= 0;y= 1
    while k < 6:
        while x >= L[k]:
            x= x-L[k];
            y= y+y/pow(10,k)
        k= k+1;
    return y*pow(10,n)

def test_exp():    
    a=-40
    b=40
    y=np.linspace(a,b,101)
    z=np.linspace(a,b,101)
    for i in range (0,101):
        z[i]=abs(exp(y[i])-exp1(y[i]))*100/(exp(y[i]))
    plt.subplot(222)
    plt.plot(y,z)
    plt.title("Erreur relative à la fonction exp")
    plt.xlabel("y")
    plt.ylabel("Erreur relative")

#-------------------------Arctan(x)--------------------------------
def atan1(x):
    if x==0:
        return 0.0
    inverse=0
    signe=0
    if x<0:
        x=-x
        signe=1
    if x>1:
        x=1/x
        inverse=1
    k= 0; y= 1; r= 0
    while (k <= 4):
        while (x <= y/pow(10,k)):
            k= k+1
        if k<=4:
            xp= x-y/pow(10,k)
            y= y+x/pow(10,k)
            x= xp
            r= r+A[k]
   
    if inverse==1:
        r=pi/2-r
    if signe==1:
        r=-r
    return r

def test_atan():    
    a=-20
    b=20
    y=np.linspace(a,b,101)
    z=np.linspace(a,b,101)
    t=np.linspace(a,b,101)
    for i in range (0,101):
        if (atan(y[i])==0):
            z[i]=0
        else :
            z[i]=100*abs(atan1(y[i])-atan(y[i]))/abs(atan(y[i]))
    plt.subplot(223)
    plt.plot(y,z)
    plt.title("Erreur relative à la fonction atan")
    plt.xlabel("y")
    plt.ylabel("Erreur relative")

#-------------------------Tan(x)--------------------------------    
def tan1(x):
    x=x%pi
    s=0
    t=0
    if x>pi/2:
        x=pi-x
        s=1
    if x>pi/4:
        x=pi/2-x
        t=1
    k= 0;n= 0;d= 1;
    while k <= 4:
        while x >= A[k]:
            x= x-A[k]           
            np= n+d/pow(10,k)
            d= d-n/pow(10,k)
            n= np
        k= k+1
    rslt=(n+x*d)/(d-x*n)
    if t==1:
        rslt=1/rslt
    if s==1:
        rslt=-rslt
    return rslt

def test_tan():
    a=-1.5
    b=1.5
    y=np.linspace(a,b,101)
    z=np.linspace(a,b,101)
    t=np.linspace(a,b,101)
    for i in range (0,101):
        if (tan(y[i])==0):
            z[i]=0
        else :
            z[i]=100*abs(tan1(y[i])-tan(y[i]))/abs(tan(y[i]))
    plt.subplot(224)
    plt.plot(y,z)
    plt.title("Erreur relative à la fonction tan")
    plt.xlabel("y")
    plt.ylabel("Erreur relative")
    plt.show()
test_ln()
test_exp()
test_atan()    
test_tan()
