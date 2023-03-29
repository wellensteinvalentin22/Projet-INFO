#coucou 
#Importation
import math as ma

#Définition des variables
M = 140000 #en kg, pour une rame
g = 9,81 
alpha = [0 for e in range(500)]

#Calcul de la vitesse
#Hypothèse : profil de vitesse trapézoïdal, vitesse max de 60 km/h atteinte au 1er km. On freine au 4ème km.
V=[]
for i in range(100):
    V.append(0,6*i)
for i in range(300):
    V.append(60)
for i in range(400,500):
    V.append(60-0,6*i)

#Calcul Puissance Train
def Puissance_Train ():
    P = []
    for i in range(500):
        P.append((M*g*ma.sin(alpha[i])+resistance_mouvement[i])*V[i])
    return(P)

def resistance_mouvement(V):
    a=0.3
    b=0.1
    l=[]
    for i in range(len(V)):
        l.append(1000+ a*V[i] + b*V[i]**2)
    return l

def force_freinage(a,V): #vitesse definie
    l=[0 for i in range(V)]
    for i in range(400,len(V)):
        l[i]=M*a+resistance_mouvement(V[i])
    return l


#Définition des variables pour le circuit électrique

Is1 = #courant provenant du cable venant de la sous-station (SS) 1
Is2 = #courant provenant du cable venant de la SS2
Rs1 = #résistance interne des SS
Rs2 = Rs1
Rl = 0,016 * 10**(-3)#résistance linéique cable entre SS1 et train (en Ohm par m)
Ud0 = 835 #en V
L = 5000 #en m, distance entre chaque sous-station

#Calcul Puissance Électrique

def Puissance_Electrique (Is1, Is2, position):
    U_train = Ud0-Is2/(Rs2 + Rl * position)
    I_train = Is1 + Is2
    return U_train * I_train