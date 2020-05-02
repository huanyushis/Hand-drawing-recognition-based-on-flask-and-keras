import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import random
def get_data():
    path = r'C:\Users\admin\Desktop\123'
    sum = 0
    datas = []
    labels = []
    classes = []
    lists = os.listdir(path)
    for i, j in enumerate(lists):
        data = np.load(path + "\\%s" % (j)).astype('uint8')
        datas.extend(data)
        classes.append(j[18:-4])
        labels.extend([i]*data.shape[0])

    datas = np.array(datas).reshape(-1, 28, 28, 1)
    labels = np.array(labels)
    return datas, labels, classes

train_x, train_y, classes = get_data()
train_x = train_x.astype('float32')
train_x /= 255
train_y = to_categorical(train_y, len(classes))
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.1,random_state=1)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.15,random_state=1)

inputs = Input(shape=(28, 28, 1))
outputs1 = Conv2D(64, (3, 3), activation='relu')(inputs)
drop1=Dropout(0.5)(outputs1)
outputs2 = Conv2D(256, (3, 3), activation='relu')(drop1)
maxpool1 = MaxPool2D((2, 2))(outputs2)
outputs3 = Conv2D(256, (3, 3), activation='relu')(maxpool1)
drop2=Dropout(0.5)(outputs3)
outputs4 = Conv2D(64, (3, 3), activation='relu')(drop2)
maxpool2 = MaxPool2D((2, 2))(outputs4)
flatten = Flatten()(maxpool2)
outputs = Dense(len(classes), activation='softmax')(flatten)

model = Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(),
              metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath="ep{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}-val_acc{acc:.3f}.h5",
                             monitor='val_acc',
                             verbose=1,
                             save_best_only='True',
                             mode='max',
                             period=1)
batch_size = 1000
epochs = 200
model.fit(train_x, train_y,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(val_x, val_y),
          callbacks=[checkpoint],
          verbose=1)
model.save('123.h5')
