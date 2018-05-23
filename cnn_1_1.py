import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

train = "train.csv"
#test = "../input/test.csv"
train = pd.read_csv(train)
#test = pd.read_csv(test)

print(train.isnull().any().sum())
#print(test.isnull().any().sum())

y= np.array(train["label"].values)
X= np.array(train.drop(labels = ["label"],axis = 1) )

#del train
#del test

x_train, x_v, y_train, y_v= train_test_split(X, y, test_size= 0.2, random_state= 14)

x_train= (x_train/255).reshape(-1, 28, 28, 1)

x_v= (x_v/255).reshape(-1, 28, 28, 1)

y_train =to_categorical(y_train)

y_v =to_categorical(y_v)

model= Sequential()
model.add(Conv2D(filters= 16, kernel_size= (5, 5), activation='relu', input_shape = (28, 28, 1)))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters= 32, kernel_size= (5, 5), activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer= 'adam', metrics= ['accuracy'])
datagen = ImageDataGenerator(zoom_range = 0.2, rotation_range = 10)


cb= LearningRateScheduler(lambda x: 0.01*0.99**x)
hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=16), steps_per_epoch=500,
                           epochs= 50, verbose=2, validation_data=(x_v[:200,:], y_v[:200,:]),
                           callbacks=[cb])
l, score= model.evaluate(x_v, y_v)
print("Loss: ", l)
print("Accuracy: ", score)

y_pred= model.predict(x_v)
y_pred = np.argmax(y_pred, axis=1)
y_v= np.argmax(y_v, axis=1)
cm= confusion_matrix(y_v, y_pred)
print(cm)




'''
Using 50 epochs:
0
Epoch 1/50
 - 7s - loss: 0.4506 - acc: 0.8582 - val_loss: 0.1807 - val_acc: 0.9500
Epoch 2/50
 - 7s - loss: 0.3074 - acc: 0.9121 - val_loss: 0.2568 - val_acc: 0.9450
Epoch 3/50
 - 7s - loss: 0.2631 - acc: 0.9277 - val_loss: 0.1196 - val_acc: 0.9650
Epoch 4/50
 - 7s - loss: 0.2458 - acc: 0.9318 - val_loss: 0.1513 - val_acc: 0.9650
Epoch 5/50
 - 7s - loss: 0.2767 - acc: 0.9263 - val_loss: 0.1609 - val_acc: 0.9550
Epoch 6/50
 - 8s - loss: 0.2575 - acc: 0.9323 - val_loss: 0.1453 - val_acc: 0.9750
Epoch 7/50
 - 7s - loss: 0.2613 - acc: 0.9346 - val_loss: 0.2684 - val_acc: 0.9500
Epoch 8/50
 - 7s - loss: 0.2624 - acc: 0.9361 - val_loss: 0.2900 - val_acc: 0.9450
Epoch 9/50
 - 7s - loss: 0.2647 - acc: 0.9330 - val_loss: 0.1718 - val_acc: 0.9500
Epoch 10/50
 - 7s - loss: 0.2380 - acc: 0.9405 - val_loss: 0.1362 - val_acc: 0.9650
Epoch 11/50
 - 7s - loss: 0.2466 - acc: 0.9357 - val_loss: 0.1035 - val_acc: 0.9650
Epoch 12/50
 - 7s - loss: 0.2465 - acc: 0.9404 - val_loss: 0.1133 - val_acc: 0.9700
Epoch 13/50
 - 7s - loss: 0.2574 - acc: 0.9365 - val_loss: 0.1960 - val_acc: 0.9450
Epoch 14/50
 - 7s - loss: 0.2670 - acc: 0.9380 - val_loss: 0.1347 - val_acc: 0.9750
Epoch 15/50
 - 7s - loss: 0.2725 - acc: 0.9355 - val_loss: 0.0898 - val_acc: 0.9700
Epoch 16/50
 - 7s - loss: 0.2677 - acc: 0.9382 - val_loss: 0.0948 - val_acc: 0.9650
Epoch 17/50
 - 7s - loss: 0.2127 - acc: 0.9465 - val_loss: 0.1112 - val_acc: 0.9750
Epoch 18/50
 - 7s - loss: 0.2387 - acc: 0.9430 - val_loss: 0.1351 - val_acc: 0.9650
Epoch 19/50
 - 7s - loss: 0.2405 - acc: 0.9477 - val_loss: 0.1698 - val_acc: 0.9800
Epoch 20/50
 - 7s - loss: 0.2381 - acc: 0.9446 - val_loss: 0.1140 - val_acc: 0.9650
Epoch 21/50
 - 7s - loss: 0.2384 - acc: 0.9403 - val_loss: 0.2500 - val_acc: 0.9600
Epoch 22/50
 - 7s - loss: 0.2333 - acc: 0.9436 - val_loss: 0.1370 - val_acc: 0.9700
Epoch 23/50
 - 7s - loss: 0.2369 - acc: 0.9491 - val_loss: 0.2119 - val_acc: 0.9700
Epoch 24/50
 - 7s - loss: 0.2482 - acc: 0.9447 - val_loss: 0.2255 - val_acc: 0.9600
Epoch 25/50
 - 7s - loss: 0.2567 - acc: 0.9427 - val_loss: 0.2200 - val_acc: 0.9550
Epoch 26/50
 - 7s - loss: 0.2346 - acc: 0.9469 - val_loss: 0.1809 - val_acc: 0.9700
Epoch 27/50
 - 7s - loss: 0.2029 - acc: 0.9513 - val_loss: 0.1249 - val_acc: 0.9650
Epoch 28/50
 - 7s - loss: 0.2572 - acc: 0.9461 - val_loss: 0.1273 - val_acc: 0.9750
Epoch 29/50
 - 7s - loss: 0.2352 - acc: 0.9463 - val_loss: 0.1801 - val_acc: 0.9600
Epoch 30/50
 - 7s - loss: 0.2354 - acc: 0.9480 - val_loss: 0.1824 - val_acc: 0.9550
Epoch 31/50
 - 7s - loss: 0.2235 - acc: 0.9526 - val_loss: 0.1695 - val_acc: 0.9700
Epoch 32/50
 - 7s - loss: 0.2189 - acc: 0.9519 - val_loss: 0.0910 - val_acc: 0.9700
Epoch 33/50
 - 7s - loss: 0.2878 - acc: 0.9451 - val_loss: 0.2084 - val_acc: 0.9550
Epoch 34/50
 - 7s - loss: 0.2344 - acc: 0.9477 - val_loss: 0.1501 - val_acc: 0.9800
Epoch 35/50
 - 7s - loss: 0.2255 - acc: 0.9494 - val_loss: 0.1126 - val_acc: 0.9900
Epoch 36/50
 - 7s - loss: 0.2655 - acc: 0.9475 - val_loss: 0.2537 - val_acc: 0.9500
Epoch 37/50
 - 7s - loss: 0.2509 - acc: 0.9482 - val_loss: 0.0585 - val_acc: 0.9900
Epoch 38/50
 - 7s - loss: 0.2170 - acc: 0.9536 - val_loss: 0.0930 - val_acc: 0.9800
Epoch 39/50
 - 7s - loss: 0.2077 - acc: 0.9579 - val_loss: 0.1366 - val_acc: 0.9700
Epoch 40/50
 - 7s - loss: 0.2368 - acc: 0.9504 - val_loss: 0.4611 - val_acc: 0.9400
Epoch 41/50
 - 7s - loss: 0.2353 - acc: 0.9515 - val_loss: 0.2073 - val_acc: 0.9750
Epoch 42/50
 - 7s - loss: 0.2555 - acc: 0.9445 - val_loss: 0.0907 - val_acc: 0.9750
Epoch 43/50
 - 7s - loss: 0.1970 - acc: 0.9514 - val_loss: 0.1778 - val_acc: 0.9650
Epoch 44/50
 - 7s - loss: 0.2430 - acc: 0.9509 - val_loss: 0.1726 - val_acc: 0.9700
Epoch 45/50
 - 7s - loss: 0.2254 - acc: 0.9549 - val_loss: 0.1489 - val_acc: 0.9700
Epoch 46/50
 - 7s - loss: 0.2106 - acc: 0.9552 - val_loss: 0.1698 - val_acc: 0.9650
Epoch 47/50
 - 7s - loss: 0.2243 - acc: 0.9520 - val_loss: 0.0782 - val_acc: 0.9750
Epoch 48/50
 - 7s - loss: 0.2371 - acc: 0.9566 - val_loss: 0.1733 - val_acc: 0.9750
Epoch 49/50
 - 7s - loss: 0.2120 - acc: 0.9572 - val_loss: 0.0815 - val_acc: 0.9800
Epoch 50/50
 - 7s - loss: 0.1818 - acc: 0.9599 - val_loss: 0.0968 - val_acc: 0.9800
8400/8400 [==============================] - 3s 310us/step
Loss:  0.10625049267
Accuracy:  0.976904761905
[[807   0   0   0   0   1   3   1   6   1]
 [  0 900   7   0   0   1   1   5  10   0]
 [  3   1 800   0   2   0   0  12   4   0]
 [  0   0   5 825   0   1   0   2   8   3]
 [  0   3   0   0 786   0   4   0   0  11]
 [  2   2   0   6   0 763   4   1   5   7]
 [  3   0   2   0   1   5 836   0   7   0]
 [  0   1   6   1   4   0   0 839   1   7]
 [  0   0   3   3   1   3   0   0 819   7]
 [  1   0   0   2   7   0   0   4   3 831]]
'''