#coucou 

#Définition des variables

#Calcul Puissance Train
def Puissance_Train (V):
    pass

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


#Calcul Puissance Électrique
