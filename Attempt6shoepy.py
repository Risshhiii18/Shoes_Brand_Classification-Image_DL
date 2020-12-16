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

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train, test_size = 0.3, random_state=0)

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

def encoder(input_img):
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(number_of_classes, activation='softmax')(den)
    return out

encode = encoder(input_img)
full_model = Model(input_img,fc(encode))

full_model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])
full_model.summary()
classify_train = full_model.fit(X_train, Y_train, batch_size=64,epochs=50,verbose=1,validation_data=(X_train, Y_train))
full_model.save('C:/Users/chuda/Bootcamp_project/Deep Learing/Level2_Problem2.model')
accuracy = classify_train.history['accuracy']
val_accuracy = classify_train.history['val_accuracy']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

test_eval = full_model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

predicted_classes = full_model.predict(x_test)
predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
print(predicted_classes)
predicted_classes.shape, y_test.shape
##########################################################################################
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

pred = full_model.predict(Xt)
pred = np.argmax(np.round(pred),axis=1)
print(pred)
import pandas as pd
zt1 = pd.Series(zt).str.replace(".jpg","")
predict = pd.DataFrame(pred, index=zt1)
plt.hist(predict[0],3)
predict.to_csv('C:/Users/chuda/Bootcamp_project/Deep Learing/Level2_Problem2/Attempt6shoe.csv')
