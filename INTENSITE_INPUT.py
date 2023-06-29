##Implémentation du courant d'accélération
I_train_acc = []
stop1, stop2, stop3 = 0,0,0
for i in range(len(V_t)) :
    if V_t[i] > 10 :
        stop1 = i-1
        break
for i in range(len(V_t)) :
    if V_t[i] > 14:
        stop2 = i-1
        break
for i in range(len(V_t)) :
    if V_t[i] >= 20 :
        stop3 = i
        break
print(stop1,stop2,stop3)

for i in range (stop3):
    v = V_t[i]
    if v < 10:
        I_train_acc.append(100+900*i/62)
    if 10 <= v < 14:
        I_train_acc.append(1000)
    if 14 <= v <= 20:
        I_train_acc.append(1000-400*i/249)


print(len(I_train_acc))


##Implémentation du courant de freinage
I_train_freinage = []
stop4 = 0
INDICE_FREINAGE = 0


for i in range(len(V_t)//2,len(V_t)) :
    if V_t[i] < 14 :
        stop4 = i-1
        break

for i in range(D):
    if X[i]>(D-d_freinage):
        INDICE_FREINAGE = i-1
        break

for i in range (INDICE_FREINAGE,len(V_t)):
    v = V_t[i]
    if v > 14:
        I_train_freinage.append(-1000*i/stop4)
    if v <= 14:
        I_train_freinage.append(-1000)

print(len(I_train_freinage))