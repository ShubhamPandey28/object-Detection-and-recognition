import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

data = keras.datasets.fashion_mnist
(train_x,train_y),(test_x,test_y)= data.load_data()

test_x = test_x/255
train_x = train_x/255

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def build_model(node1,node2,optimizer,train_x,train_y):
    model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),keras.layers.Dense(node1,activation=tf.nn.relu),keras.layers.Dense(node2,activation=tf.nn.softmax)])
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_x,train_y,epochs=5)
    return model

model = build_model(128,10,'adam',train_x,train_y)

loss, acc = model.evaluate(test_x,test_y)

predict_y = model.predict(test_x)

i = np.random.choice(list(range(len(test_y))))
p_y = np.argmax(predict_y[i])
print(labels[p_y],round(100*predict_y[i][p_y],2),labels[test_y[i]])
