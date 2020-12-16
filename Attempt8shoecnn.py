import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.optimizers import Adam,RMSprop
from keras.utils import np_utils
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
DATADIR = "C:/Users/chuda/Bootcamp_project/Deep Learing/Level2_Problem2/Data/train"
CATEGORIES = ['bata','clarks','leecooper']
IMG_SIZE = 28

training_data = []

def create_training_data():
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)

        for img in (os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
            
create_training_data()
print(len(training_data))

X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)
plt.hist(y, 3)

X_train = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE,1)
X_train = X_train/255.0
number_of_classes = 3
Y_train = np_utils.to_categorical(y,number_of_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(5, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(number_of_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
model.fit(X_train, Y_train, validation_data=(X_train, Y_train),shuffle=True, epochs=250, batch_size=50)

test_eval = model.evaluate(X_train, Y_train, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])
##########################################
test_data = []
DAT = "C:/Users/chuda/Bootcamp_project/Deep Learing/Level2_Problem2/Data/test"
CAT = ['bata','clarks','leecooper']

def new_data():
    for category in CAT:

        path = os.path.join(DAT,category)

        for img in (os.listdir(path)):
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            test_data.append([new_array, category,img])

new_data()


Xt = []
yt =[]
zt = []

for features,lab,name in test_data:
    Xt.append(features)
    yt.append(lab)
    zt.append(name)

Xt = np.array(Xt).reshape(-1, IMG_SIZE, IMG_SIZE,1)

Xt = Xt/255.0

pred = model.predict(Xt)
pred = np.argmax(np.round(pred),axis=1)
print(pred)
import pandas as pd
zt1 = pd.Series(zt).str.replace(".jpg","")
predict = pd.DataFrame(pred, index=zt1)
plt.hist(predict[0],3)
predict.to_csv('C:/Users/chuda/Bootcamp_project/Deep Learing/Level2_Problem2/Attempt8shoe.csv')