# regulizer

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.regularizers import l2,l1


#dense = couche entiérement connecté, fait sur la couche de sortie
#load _ data retourne variables x,y de test et de train
#database s'appele mnist 
 
## 1.classifier la BD
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(np.shape(x_train),np.shape(y_train))
print(np.shape(x_test),np.shape(y_test))

#x(60000 nb images d'apprentissage 28,28 taille d'image)et y(60000,) nb d'étiquette correspondant
#1000 nb image de test 
#profondeur par défaut 1

\n
## 2.visualiser qlq expl d'apprentissage

plt.rcParams["figure.figsize"]=[9.,9.] #taille de l'image
plt.rcParams["figure.autolayout"]=False

fig,axs = plt.subplots(nrows=3, ncols=3)
axs = axs.flatten()
for ax,j in zip(axs,range(9)):
  ax.axis("off")
  ax.imshow(x_train[j], cmap="gray")
  ax.set_title(str(y_train[j]))
plt.show()



## 3. préparer les données en pixels (applatissement + normalisation)
x_train_v=np.reshape(x_train, [60000,784])/255  # la val doit etre entre 0 et 255 dnc pour avoir normalisation on divise sur 60000 images ayant 28*28=784pixels sur 255 
x_test_v=np.reshape(x_test, [10000,784])/255

print(np.shape(x_train_v))
print(np.shape(x_test_v))

## 4.catégoriser la cible (one hot encoding)

from tensorflow.keras.utils import to_categorical
y_train_c=to_categorical(y_train)
y_test_c=to_categorical(y_test)

print("avant", np.shape(y_train), "après", np.shape(y_train_c))
print("avant", y_train[10])
print("avant", y_train_c[10])





## 5.1développer un modèle RN avec 0 couche caché
#les parametres de modeles sont omégas wi et les biais bi
def perceptron_monocouche():
  inp=Input(shape=(784,))
  outp=Dense(10,activation="softmax") (inp) # (dense pour les paramètres de modèles et inp entré de cette couche n)
  model = Model (inp,outp)
  model.summary()
  return model
  
  
  
  
  
  ## 5.2 avec multicouches
def perceptron_multicouche():
  inp=Input(shape=(784,)) #784 pixel avec , étiquette 
  c1=Dense(200,activation="sigmoid") (inp) 
  c2=Dense(100,activation="sigmoid") (c1) 
  c3=Dense(60,activation="sigmoid") (c2) 
  c4=Dense(30,activation="sigmoid") (c3) 
  outp=Dense(10,activation="softmax") (c4) 
  model = Model (inp,outp)
  model.summary()
  return model

  #ns avons utiliser sigmoide dans c1 à c4 psk ns sommes pas besoin de comparer ci par rapport à une autre cj
  
  
  
  ## 5.2 avec multicouches
def perceptron_multicouche_relu():
  inp=Input(shape=(784,)) #784 pixel avec , étiquette 
  c1=Dense(200,activation="relu") (inp) #les valeurs de neurons pour chaque couche est donné dans l'exercice
  c2=Dense(100,activation="relu") (c1) 
  c3=Dense(60,activation="relu") (c2) 
  c4=Dense(30,activation="relu") (c3) 
  outp=Dense(10,activation="softmax") (c4) 
  model = Model (inp,outp)
  model.summary()
  return model
  
  
  
  ## 6.model d'entrainement
#a.fct cout
fc="categorical_crossentropy" #psk on a plusieurs classes si on a que 2 on va utiliser binary_crossentropy

#fct d'optimisation SGD
optim=tf.keras.optimizers.SGD(learning_rate=0.001)





## c. apprentissage model monocouche
from tensorflow.python import metrics
modelA=perceptron_monocouche()
modelA.compile(loss=fc, optimizer=optim, metrics=["accuracy"])
#on utilise le batch pour lot pour éviter plusieur problème, si on utilise les données 1 par 1 on va se tarder, si on utilise tous a la fois il va consommer la mémoire
#fit pour apprendre 
histA=modelA.fit(x_train_v,y_train_c, validation_data=(x_test_v,y_test_c), epochs=100, batch_size=100, shuffle=True)

## c. apprentissage model multicouche (sigmoid)
modelB=perceptron_multicouche()
modelB.compile(loss=fc, optimizer=optim, metrics=["accuracy"])
histB=modelB.fit(x_train_v,y_train_c, validation_data=(x_test_v,y_test_c), epochs=100, batch_size=100, shuffle=True)


## 7.visualiser les couches
plt.rcParams["figure.figsize"]=[6., 5.]
plt.rcParams.update()
plt.show()


## 8.evaluer le modèles sur ensemble de test
test_loss_modelB, test_acc_modelB = modelB.evaluate(x_test_v,  y_test_c, verbose=2)





## modèle de base (multicouche, relu, softmax)
## c. apprentissage
modelC=perceptron_multicouche_relu()
modelC.compile(loss=fc, optimizer=optim, metrics=["accuracy"])
histC=modelC.fit(x_train_v,y_train_c, validation_data=(x_test_v,y_test_c), epochs=100, batch_size=100, shuffle=True)

## 7.visualiser les erreurs
plt.rcParams["figure.figsize"] = [6., 5.]
plt.rcParams.update({'font.size': 14})
plt.plot(histC.history["accuracy"],c='b',label="Ens. d'apprentissage")
plt.plot(histC.history["val_accuracy"],c='r',label="Ens. de test")
plt.xlabel("n_epoch")
plt.ylabel("Taux de classification")
plt.legend()
plt.show()


## 8. Evaluer vos modèle sur l’ensemble de test
plt.rcParams["figure.figsize"] = [9., 9.]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(nrows=3, ncols=3)
axs = axs.flatten()
y_pred=modelC.predict(x_test_v)
color="red"
for ax,j in zip(axs, range(9)):
  ax.axis("off")
  ax.imshow(x_test[j], cmap='gray')
if (y_test[j]==np.argmax(y_pred[j])):
  color="green"
else:
  color="red"

ax.set_title("V.Réelle:"+str(y_test[j])+"Prédiction:"+str(np.argmax(y_pred[j])),c=color)
plt.show()




indice_exemple=1
image_test1=x_test_v[indice_exemple]
predictionProba =modelC.predict(image_test1.reshape(1,-1))
predictionClass=np.argmax(predictionProba)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.imshow(x_test[indice_exemple], cmap='gray')
ax1.axis("off")
y_pos = np.arange(10)
barlist=ax2.barh(y_pos,predictionProba[0])
if(predictionClass==y_test[indice_exemple]):
  barlist[int(predictionClass)].set_color('g')
else:
  barlist[int(predictionClass)].set_color('r')
plt.yticks(y_pos, y_pos, fontsize=20)
plt.show()












## 8.evaluer le modèles sur ensemble de test
test_loss_modelC, test_acc_modelC = modelC.evaluate(x_test_v,  y_test_c, verbose=2)
pred_C= modelC.predict(x_test_v) 
pred_C= np.argmax(pred_C, axis = 1)[:50] 
label_C= np.argmax(y_test_c,axis = 1)[:50] 

print(pred_C) 
print(label_C)



# Lasso (L1) rgulizer
def lasso_reg(lambd):
  inp=Input(shape=(784,)) #784 pixel avec , étiquette 
  c1=Dense(200, kernel_regularizer=l1(lambd),activation="relu") (inp) #les valeurs de neurons pour chaque couche est donné dans l'exercice
  c2=Dense(100, kernel_regularizer=l1(lambd),activation="relu") (c1) 
  c3=Dense(60, kernel_regularizer=l1(lambd),activation="relu") (c2) 
  c4=Dense(30, kernel_regularizer=l1(lambd),activation="relu") (c3) 
  outp=Dense(10,activation="softmax") (c4) 
  model = Model (inp,outp)
  model.summary()
  return model
  
## c.apprentissage
optimAd=tf.keras.optimizers.Adam(learning_rate=0.01)
lasso=lasso_reg(lambd=0.001)
lasso.compile(loss=fc, optimizer=optimAd,metrics=["accuracy"])
hist_lasso=lasso.fit(x_train_v,y_train_c, validation_data=(x_test_v,y_test_c), epochs=100, batch_size=100, shuffle=True)

## 7.visualiser les erreurs
plt.rcParams["figure.figsize"] = [6., 5.]
plt.rcParams.update({'font.size': 14})
plt.plot(hist_lasso.history["accuracy"],c='b',label="Ens. d'apprentissage")
plt.plot(hist_lasso.history["val_accuracy"],c='r',label="Ens. de test")
plt.xlabel("n_epoch")
plt.ylabel("Taux de classification")
plt.legend()
plt.show()


## 8. Evaluer vos modèle sur l’ensemble de test
plt.rcParams["figure.figsize"] = [9., 9.]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(nrows=3, ncols=3)
axs = axs.flatten()
y_pred_lasso=lasso.predict(x_test_v)
color="red"
for ax,j in zip(axs, range(9)):
  ax.axis("off")
  ax.imshow(x_test[j], cmap='gray')
if (y_test[j]==np.argmax(y_pred_lasso[j])):
  color="green"
else:
  color="red"

ax.set_title("V.Réelle:"+str(y_test[j])+"Prédiction:"+str(np.argmax(y_pred_lasso[j])),c=color)
plt.show()


indice_exemple=1
image_test1=x_test_v[indice_exemple]
predictionProba =lasso.predict(image_test1.reshape(1,-1))
predictionClass=np.argmax(predictionProba)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.imshow(x_test[indice_exemple], cmap='gray')
ax1.axis("off")
y_pos = np.arange(10)
barlist=ax2.barh(y_pos,predictionProba[0])
if(predictionClass==y_test[indice_exemple]):
  barlist[int(predictionClass)].set_color('g')
else:
  barlist[int(predictionClass)].set_color('r')
plt.yticks(y_pos, y_pos, fontsize=20)
plt.show()

## 9.evaluer le modele de test
test_loss_lasso, test_acc_lasso = lasso.evaluate(x_test_v,  y_test_c, verbose=2)

# Ridge (L2) regulizer
## definitin d'un modele
def ridge_reg(lambd):
  inp=Input(shape=(784,)) #784 pixel avec , étiquette 
  c1=Dense(200, kernel_regularizer=l2(lambd),activation="relu") (inp) #les valeurs de neurons pour chaque couche est donné dans l'exercice
  c2=Dense(100, kernel_regularizer=l2(lambd),activation="relu") (c1) 
  c3=Dense(60, kernel_regularizer=l2(lambd),activation="relu") (c2) 
  c4=Dense(30, kernel_regularizer=l2(lambd),activation="relu") (c3) 
  outp=Dense(10,activation="softmax") (c4) 
  model = Model (inp,outp)
  model.summary()
  return model
  
 ## phase d'apprentissage
 ridge=ridge_reg(lambd=0.0001)
 ridge.compile(loss=fc, optimizer=optimAd,metrics=["accuracy"])
 hist_ridge=ridge.fit(x_train_v,y_train_c, validation_data=(x_test_v,y_test_c), epochs=100, batch_size=100, shuffle=True)
 
 
 ## visualiser les erreurs
plt.rcParams["figure.figsize"] = [6., 5.]
plt.rcParams.update({'font.size': 14})
plt.plot(hist_ridge.history["accuracy"],c='b',label="Ens. d'apprentissage")
plt.plot(hist_ridge.history["val_accuracy"],c='r',label="Ens. de test")
plt.xlabel("n_epoch")
plt.ylabel("Taux de classification")
plt.legend()
plt.show()


## 8. Evaluer vos modèle sur l’ensemble de test

plt.rcParams["figure.figsize"] = [8., 8.]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(nrows=3, ncols=3)
axs = axs.flatten()
y_pred_ridge=ridge.predict(x_test_v)
color="red"
for ax,j in zip(axs, range(9)):
  ax.axis("off")
  ax.imshow(x_test[j], cmap='gray')
if (y_test[j]==np.argmax(y_pred_ridge[j])):
  color="green"
else:
  color="red"

ax.set_title("V.Réelle:"+str(y_test[j])+"Prédiction:"+str(np.argmax(y_pred_ridge[j])),c=color)
plt.show()


## evaluer modele de test 
test_loss_ridge, test_acc_ridge= ridge.evaluate(x_test_v,  y_test_c, verbose=2)


# weight decay regulizer
## fct d'optimisation SGD
optim=tf.keras.optimizers.SGD(learning_rate=0.001)
wd=ridge_reg(lambd=0.0001)
wd.compile(loss=fc, optimizer=optim,metrics=["accuracy"])
hist_wd=wd.fit(x_train_v,y_train_c, validation_data=(x_test_v,y_test_c), epochs=100, batch_size=100, shuffle=True)
  
  
  ## 7.visualiser les erreurs
plt.rcParams["figure.figsize"] = [6., 5.]
plt.rcParams.update({'font.size': 14})
plt.plot(hist_wd.history["accuracy"],c='b',label="Ens. d'apprentissage")
plt.plot(hist_wd.history["val_accuracy"],c='r',label="Ens. de test")
plt.xlabel("n_epoch")
plt.ylabel("Taux de classification")
plt.legend()
plt.show()

## 8. Evaluer vos modèle sur l’ensemble de test
plt.rcParams["figure.figsize"] = [9., 9.]
plt.rcParams["figure.autolayout"] = True
plt.rcParams.update({'font.size': 12})
fig, axs = plt.subplots(nrows=3, ncols=3)
axs = axs.flatten()
y_pred_wd=wd.predict(x_test_v)
color="red"
for ax,j in zip(axs, range(9)):
  ax.axis("off")
  ax.imshow(x_test[j], cmap='gray')
if (y_test[j]==np.argmax(y_pred_wd[j])):
  color="green"
else:
  color="red"

ax.set_title("V.Réelle:"+str(y_test[j])+"Prédiction:"+str(np.argmax(y_pred_wd[j])),c=color)
plt.show()



indice_exemple=1
image_test1=x_test_v[indice_exemple]
predictionProba =wd.predict(image_test1.reshape(1,-1))
predictionClass=np.argmax(predictionProba)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.imshow(x_test[indice_exemple], cmap='gray')
ax1.axis("off")
y_pos = np.arange(10)
barlist=ax2.barh(y_pos,predictionProba[0])
if(predictionClass==y_test[indice_exemple]):
  barlist[int(predictionClass)].set_color('g')
else:
  barlist[int(predictionClass)].set_color('r')
plt.yticks(y_pos, y_pos, fontsize=20)
plt.show()



## evaaluer le modele de test 
test_loss_wd, test_acc_wd= wd.evaluate(x_test_v,  y_test_c, verbose=2)

## verifier estimation de donnée
pred_wd= wd.predict(x_test_v) 
pred_wd= np.argmax(pred_wd, axis = 1)[:20] 
label_wd = np.argmax(y_test_c,axis = 1)[:20] 

print(pred_wd) 
print(label_wd)
