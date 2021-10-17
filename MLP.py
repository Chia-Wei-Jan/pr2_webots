import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error

lidar = pd.read_excel("test.xlsx", usecols="B:L")
lidar_data = np.array(lidar.values)
output = pd.read_excel("test.xlsx", usecols="M:N")
output_data = np.array(output.values)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(lidar_data)
Y = scaler.fit_transform(output_data)

print("X:", X)
print("Y:", Y)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=20, activation='relu', input_shape=[X.shape[1]]))
model.add(tf.keras.layers.Dense(units=20, activation='relu'))
model.add(tf.keras.layers.Dense(units=20, activation='relu'))
model.add(tf.keras.layers.Dense(units=20, activation='relu'))
model.add(tf.keras.layers.Dense(units=20, activation='relu'))
model.add(tf.keras.layers.Dense(units=2, activation='sigmoid'))
model.summary()

model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

history = model.fit(X, Y, epochs=10000, batch_size=8)

# testing
print("start testing")
cost = model.evaluate(X, Y)
print("test cost: {}".format(cost))

Y_pred2 = model.predict(X)
predict = scaler.inverse_transform(Y_pred2)
y_origin = scaler.inverse_transform(Y)

print("predict:", predict[:10])
print("------------------------")
print("y_origin:", y_origin[:10])
print("=======================")
print(predict)
print("------------------------")
print(y_origin)

mse = mean_squared_error(y_origin, predict)
mae = mean_absolute_error(y_origin, predict)
print("===============================")
print("MSE: ", mse)
print("MAE: ", mae)
