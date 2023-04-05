#coucou
#Importation
import math as ma

##Définition des variables##
M = 140000 #en kg, pour une rame
g = 9.81
alpha = [0 for e in range(500)]

##Calcul de la vitesse##
#Hypothèse : profil de vitesse trapézoïdal, vitesse max de 60 km/h atteinte au 100ème pas de temps. On freine au 400 pas de temps.
V=[]
for i in range(100):
    V.append(0.6/3.6*i)
for i in range(300):
    V.append(60/3.6)
for i in range(100):
    V.append(60/3.6-0.6/3.6*i)

A=[]
for i in range(100):
    A.append(0.6/(3.6**3.6*1000))
for i in range(300):
    A.append(0)
for i in range(400,500):
    A.append(-0.6/(3.6**3.6*1000))

##Calcul Puissance Train##
def frottements():
    a=0.3
    b=0.1
    l=[]
    for i in range(len(V)):
        l.append(1000+ a*V[i] + b*V[i]**2)
    return l


def Puissance_Train():
    P = []
    for i in range(1,400):
        P.append((M*A[i] + M*g*ma.sin(alpha[i]) + frottements()[i]) * V[i])
    for i in range(100):
        P.append((M*A[i] + M*g*ma.sin(alpha[i]) - frottements()[i]) * V[i])
    return(P)


##Calcul Puissance Électrique##

#Définition des variables pour le circuit électrique
Is1 = #courant provenant du cable venant de la sous-station (SS) 1
Is2 = #courant provenant du cable venant de la SS2
Rs1 = #résistance interne des SS
Rs2 = Rs1
Rl = 0.016 * 10**(-3)#résistance linéique cable entre SS1 et train (en Ohm par m)
Ud0 = 835 #en V
L = 5000 #en m, distance entre chaque sous-station

#Calcul Puissance Électrique
def Puissance_Electrique (Is1, Is2, position):
    U_train = Ud0-Is2/(Rs2 + Rl * position)
    I_train = Is1 + Is2
    return U_train * I_train

