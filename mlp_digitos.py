import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.preprocessing
import sklearn.neural_network
import sklearn.model_selection

numeros = sklearn.datasets.load_digits()
imagenes = numeros['images']  # Hay 1797 digitos representados en imagenes 8x8
n_imagenes = len(imagenes)
X = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))
Y = numeros['target']
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.5)
scaler = sklearn.preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

l=[]
f1=[]
f1_train=[]
for i in range(20):
    mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                           hidden_layer_sizes=(i+1), 
                                           max_iter=200)
    mlp.fit(X_train, Y_train)
    l.append(mlp.loss_)
    f1.append(sklearn.metrics.f1_score(Y_test, mlp.predict(X_test), average='macro'))
    f1_train.append(sklearn.metrics.f1_score(Y_train, mlp.predict(X_train), average='macro'))

x=np.arange(1,21)
plt.figure(figsize=(12,8))
plt.subplot(121)
plt.scatter(x[6],l[6],color='red',s=40)
plt.plot(x,l)
plt.grid()
plt.xlabel('Número de Neuronas')
plt.ylabel('Loss')
plt.subplot(122)
plt.scatter(x[6],f1[6],color='red',s=40)
plt.scatter(x[6],f1_train[6],color='red',s=40)
plt.plot(x,f1,label='Test')
plt.plot(x,f1_train,label='Train')
plt.legend(loc=0.0)
plt.xlabel('Número de Neuronas')
plt.ylabel('F1 score')
plt.grid()
plt.savefig("loss_f1.png")
plt.close()

mlp = sklearn.neural_network.MLPClassifier(activation='logistic', 
                                           hidden_layer_sizes=(7), 
                                           max_iter=200)
mlp.fit(X_train, Y_train)
scale = np.max(mlp.coefs_[0])
plt.figure(figsize=(12,15))
for i in range(7):
    l1_plot = plt.subplot(7, 1, i + 1)
    l1_plot.imshow(mlp.coefs_[0][:,i].reshape(8,8),cmap=plt.cm.RdBu, 
                   vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_title('Neurona ' + str(i+1))
plt.savefig('neuronas.png')
plt.close()