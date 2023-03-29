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
def resistance_mouvement():
    pass

def Puissance_Train ():
    P = []
    for i in range(500):
        P.append((M*g*ma.sin(alpha[i])+resistance_mouvement[i])*V[i])
    return(P)

#Définition des variables pour le circuit électrique

Iss1 = #courant provenant du cable venant de la SS1
Iss2 = #courant provenant du cable venant de la SS2
Ri = #résistance interne des SS
Rss1 = #résistance linéique cable entre SS1 et train (en Ohm par m)
Rss2 = 0.016 * 10**(-3) #resistance lineique cable entre train et SS2 (en Ohm par m)
Ud0 = 835 #en V
L = 5000 #en m, distance entre chaque sous-station

#Calcul Puissance Électrique

def Puissance_Electrique ():
    U_train = 
    I_train