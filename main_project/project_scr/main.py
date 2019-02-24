import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# load data
data = pd.read_csv('heart.csv')
row, column = data.shape

# data normalized
x_data = data.drop('target',axis=1)
norm_x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

training_set_x = []
training_set_y = []
testting_set_x = []
testting_set_y = []

index_list = list(range(row))
random.shuffle(index_list)

tranning_set_length = 100
for i in index_list[:tranning_set_length]:
    training_set_x.append(list(norm_x.iloc[i,:-1]))
    training_set_y.append(data.loc[i,'target'])

for i in index_list[tranning_set_length:]:
    testting_set_x.append(list(norm_x.iloc[i,:-1]))
    testting_set_y.append(data.loc[i,'target'])

# ANN
model = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(25,activation=tf.nn.relu),
    keras.layers.Dense(25,activation=tf.nn.relu),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2,activation=tf.nn.softmax),
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

training_set_x = np.array(training_set_x)
training_set_y = np.array(training_set_y)
testting_set_x = np.array(testting_set_x)
testting_set_y = np.array(testting_set_y)

m = model.fit(training_set_x,training_set_y,epochs=150,validation_split=0.2)
test_loss, test_acc = model.evaluate(testting_set_x, testting_set_y)
print('the accurancy of testing :',round(test_acc,4))

train_acc = m.history['acc']
val_acc = m.history['val_acc']

plt.plot(train_acc,label='train_acc')
plt.plot(val_acc,label='val_acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()
plt.show()