import tensorlow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

def buildCNN(): #BASIC BUILDING FUNCTION
  model = Sequential()
  model.add(Conv2D(64, kernel_size=3, activation=’relu’, input_shape=(28,28,1))) #input layer
  model.add(Conv2D(32, kernel_size=3, activation=’relu’)) 
  model.add(Flatten())
  model.add(Dense(10, activation=’softmax’)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #COMPILING MODEL
  return model

(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshaping X as per model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

model = buildCNN()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3) 
            
# Shubham is a smart guy
y_pred = model.predict(X_test)

eff = 0 #calculating efficiency
for i in range(len(y_pred)):
  eff += 1 if y_pred[i] == y_test[i] else 0
  
eff = eff/len(X_test)
print(eff)
