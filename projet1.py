# coucou

# Importation
import math as ma
import matplotlib.pyplot as plt


##Définition des variables##
M = 140000  # en kg, pour une rame
g = 9.81
alpha = [0 for e in range(500)]
tau = 1  # pas de temps


##Calcul de la vitesse##
# Hypothèse : profil de vitesse trapézoïdal, vitesse max de 60 km/h atteinte au 100ème pas de temps. On freine au 400 pas de temps.
V = []
for i in range(25):
    V.append(0.8 * tau * i)
for i in range(450):
    V.append(20)
for i in range(25):
    V.append(20 - 0.8 * tau * i)

X = []
position = 0
for i in range(500):
    position = position + V[i] * tau
    X.append(position)

A = []
for i in range(25):
    A.append(0.8)
for i in range(450):
    A.append(0)
for i in range(25):
    A.append(-0.8)

##Calcul Puissance Train##
def frottements():
    a = 0.3
    b = 0.1
    l = []
    for i in range(len(V)):
        l.append(1000 + a * V[i] + b * V[i] ** 2)
    return l

def Puissance_Train():
    P = []
    for i in range(400):
        P.append((M * A[i] + M * g * ma.sin(alpha[i]) + frottements()[i]) * V[i])
    for i in range(400,499): #Calcul Puissance récupérée par freinage
        P.append(-0.20*(0,5*M*(V[i]**2-V[i+1]**2))/tau) #On récupère 20% de l'énergie cinétique, et on prends l'accélération et la vitesse au point juste avant de freiner
    return P

P = Puissance_Train()
print(P)


##Calcul Puissance Électrique##

# Définition des variables pour le circuit électrique
Is1 = 1200  # courant provenant du cable venant de la sous-station (SS) 1
Is2 = 1200  # courant provenant du cable venant de la SS2
Rs1 = 0.1  # résistance interne des SS négligeable
Rs2 = Rs1
Rl = 0.016 * 10 ** (-3)  # résistance linéique cable entre SS1 et train (en Ohm par m)
Ud0 = 835  # en V
L = 5000  # en m, distance entre chaque sous-station
U_train = []
I_train = []
W_train = []

# Calcul Puissance Électrique
for i in range(500):
    U_train.append(Ud0 - Is2 * (Rs2 + Rl *(X[500]- X[i]))
    I_train.append(Is1 + Is2)
    W_train.append(U_train[i] * I_train[i])


##Calcul des courbes
plt.figure()
plt.plot(X, U_train)
plt.show()
plt.figure()
plt.plot(X, P)
plt.title("Puissance en fonction de la position")

plt.show()
