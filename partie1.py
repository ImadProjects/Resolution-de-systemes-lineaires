import numpy as np
import matplotlib.pyplot as plt

#------------------------------Question 1--------------------------
def rp(x,p):
    t=1
    i=0
    if x==0 :
        return 0
    
    if x>1 or x<-1 : # si mon nombre n'est pas de la forme 0,....
        # on calcul t tel que x*t soit de la forme [+-][0-9],....
        while x*t>=10 or x*t<= -10 : 
            t/=10
            i+=1
        res=round(x*t,p-1)*10**i
        
    else : # si mon nombre est de la forme 0,.....
        # on calcul t tel que x*t soit de la forme [+-][0-9],....
        while -1<=x*t<=1  :
            t*=10
            i+=1
        res = round(x,p-1+i)
        
    return res

def test_rp():
    print(rp(3.141592658,4))
    print(rp(3.141592658,6))
    print(rp(10507.1823,4))
    print(rp(10507.1823,6))
    print(rp(0.0001857563,4))
    print(rp(0.0001857563,6))

#test_rp()

#------------------------------Question 2--------------------------
def rp_add(x,y,p):
    return rp(rp(x,p)+rp(y,p),p)    
def rp_multi(x,y,p):
    return rp(rp(x,p)*rp(y,p),p)

#------------------------------Question 3--------------------------
def erreur_relative_add (x,y,p):
    return abs((x+y)-rp_add(x,y,p))/abs(x+y)
def erreur_relative_multi (x,y,p):
    if (x==0 or y==0):
        return 0 ; 
    return abs((x*y)-rp_multi(x,y,p))/abs(x*y)

#------------------------------Question 4--------------------------
def tracer(x,y)  :
    a=-y
    b=y
    y=np.linspace(a,b,201)
    z=np.linspace(0,0,201)
    t=np.linspace(0,0,201)
    for i in range (0,201):
        z[i]=erreur_relative_add(x,y[i],3)
        t[i]=erreur_relative_multi(x,y[i],3)
    plt.plot(y,z,label="Erreur relative de l'addition")
    plt.plot(y,t,label="Erreur relative de la multiplication")
    plt.title("Erreurs relatives sur l'addition et la multiplication")
    plt.legend()
    plt.xlabel("y")
    plt.ylabel("Erreur relative")
    plt.show()

#tracer(1000000,1)        Tests pour différentes valeurs de x
#tracer(100000,40)
tracer(0.000119,1)

#------------------------------Question 5--------------------------
def log_2(p):
    s=0
    signe=1
    t=10**p
    erreur=0
    for i in range (1,t):
        s+=signe/i
        signe*=-1
    return rp(s,p)

def rp_log_2(p):
    s=0
    signe=1
    t=10**p
    for i in range (1,t):
        s=rp_add(s,signe/i,p)
        signe*=-1
    return rp(s,p)
def test_log ():
    a2=rp(np.log(2),9)
    a1=rp(np.log(2),5)
    b1=rp_log_2(5)
    b2=log_2(9)
    print("l'erreur relative du calcul de log 2 en utilisant l'addition machine  sur 9 decimal est ", a2,"-",b2,"=",abs(a2-b2))
    print("l'erreur relative du calcul de log 2 en utilisant l'addition definie sur  5 decimal est ", a1,"-",b1,"=",abs(a1-b1))
#test_log()
#l'erreur relative du calcul de log 2 en utilisant l'addition machine  sur 9 decimal est  0.693147181 - 0.693147181 = 0.0
#l'erreur relative du calcul de log 2 en utilisant l'addition definie sur  5 decimal est  0.69315 - 0.69321 = 5.99999999999e-05

