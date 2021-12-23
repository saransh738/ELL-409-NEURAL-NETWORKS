import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

data_points_train = pd.read_csv('2019MT60763.csv', header = None)
data = data_points_train.values

#Training Data
train_x = data[:2500,:784].astype(float)
train_t = data[:2500,784]

#Testing Data
test_x = data[500:,:784].astype(float)
test_t = data[500:,784]

model = Sequential()
model.add(Dense(30, input_dim=784, activation='sigmoid'))
#model.add(Dense(60, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
sgd  = keras.optimizers.SGD(learning_rate=0.08)
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(train_x, train_t, epochs=150,verbose=0)
accuracy = model.evaluate(train_x, train_t,verbose=0)
print(accuracy)
accuracy = model.evaluate(test_x, test_t,verbose=0)
print(accuracy)
