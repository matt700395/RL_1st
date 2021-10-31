from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

#data load
iris = datasets.load_iris()
X = iris.data
Y= iris.target

#scaling of inputs, and one-hot encoding of outputs
scaler = StandardScaler()
X_ = scaler.fit_transform(X)
enc = OneHotEncoder()
Y_ = enc.fit_transform(Y.reshape(-1,1)).toarray()

#train-test set split
X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.2)

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

#model define
model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dense(5, activation='ReLu'))
model.add(Dense(5, activation='ReLu'))

model.compile(optimizer='adam',loss='categorical_entropy' , matrics=['accuracy'])

model.fit()

#neural network model training
h = model.fit(X_train, Y_train, epochs = 50)
score = model.evaluate(X_test, Y_test)

#problem 2
print('accuracy : %1.2f'%score[1])

from matplotlib import pyplot

#problem 3
pyplot.plot(h.history['accuracy'])

#Evaluation
#Problem 4
print(Y_test[0:5, ])
print(np.argmax(Y_test[0:5,], axis=-1))

#problem 5
print(model.predict(X_test[0:5, ]))
print(model.predict_classes((X_test[0:5, ])))