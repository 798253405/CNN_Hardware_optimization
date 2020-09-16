

from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras import backend
from sklearn.decomposition import NMF
#from tensorflow.keras.engine.topology import Layer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import datetime
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
#load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0#normalizing
print(x_train.size)












x_train=np.resize(x_train,(60000,784))
x_test=np.resize(x_test,(10000,784))
a=60000
#show one figure to see whether it is right
'''
x_show=np.resize(x_train,(-1,28,28,))
x_show1=x_show[1]
plt.imshow(x_show1, cmap='gray')
plt.show()
print(x_show1.size)

'''


x_train=x_train[0:a,:]


x_show=np.resize(x_train,(-1,28,28,))
x_show1=x_show[2000]
plt.imshow(x_show1, cmap='gray')
plt.show()
print(x_show1.size)

#x_train=np.transpose(x_train)

#x_test=np.transpose(x_test)
starttime=datetime.datetime.now()

nmf = NMF(n_components=90,random_state=0,max_iter=200)
nmf.fit(x_train)

H_train_nmf = nmf.transform(x_train)
Hh_train_nmf=nmf.components_
H_test_nmf = nmf.transform(x_test)


knn_nmf = KNeighborsClassifier(n_neighbors=5,n_jobs=12)
knn_nmf.fit(H_train_nmf,y_train[0:a])
print('Train score:',knn_nmf.score(H_train_nmf,y_train[0:a]))
print('test score:',knn_nmf.score(H_test_nmf,y_test))
oneruntime = datetime.datetime.now()
print((oneruntime - starttime).seconds, 'time')
