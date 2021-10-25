import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pydot
import graphviz
from sklearn.model_selection import KFold

lidar = pd.read_excel("test.xlsx", usecols="B:L")
lidar_data = np.array(lidar.values)
output = pd.read_excel("test.xlsx", usecols="M:N")
output_data = np.array(output.values)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(lidar_data)
Y = scaler.fit_transform(output_data)

kf = KFold(n_splits=10, shuffle=True, random_state=False)

cost_arr = []
mse_arr = []
mae_arr = []
predict_arr = np.zeros([Y.shape[0], Y.shape[1]])

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    inputs = keras.Input(shape=(11,))
    hA_1 = layers.Dense(20, activation="relu")(inputs)
    hB_1 = layers.Dense(20, activation="relu")(hA_1)
    hC_1 = layers.Dense(20, activation="relu")(hA_1)
    average1 = layers.Average()([hA_1, hB_1, hC_1])
    hA_2 = layers.Dense(20, activation="relu")(average1)
    hB_2 = layers.Dense(20, activation="relu")(hA_2)
    hC_2 = layers.Dense(20, activation="relu")(hA_2)
    average2 = layers.Average()([hA_2, hB_2, hC_2])
    hA_3 = layers.Dense(20, activation="relu")(average2)
    hB_3 = layers.Dense(20, activation="relu")(hA_3)
    hC_3 = layers.Dense(20, activation="relu")(hA_3)
    average3 = layers.Average()([hA_3, hB_3, hC_3])
    hA_4 = layers.Dense(20, activation="relu")(average3)
    outputs = layers.Dense(2, activation='sigmoid')(hA_4)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    keras.utils.plot_model(model, to_file='model2.png', show_shapes=True)
    model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

    history = model.fit(X_train, Y_train, epochs=10000, batch_size=8)

    # testing
    print("start testing")
    cost = model.evaluate(X_test, Y_test)
    cost_arr.append(cost)
    print("test cost: {}".format(cost))

    Y_pred2 = model.predict(X_test)
    predict = scaler.inverse_transform(Y_pred2)
    y_origin = scaler.inverse_transform(Y_test)

    print(predict)
    print("------------------------")
    print(y_origin)

    predict_arr[test_index] = predict

    mse = mean_squared_error(y_origin, predict)
    mse_arr.append(mse)

    mae = mean_absolute_error(y_origin, predict)
    mae_arr.append(mae)
    print("===============================")
    print("MSE: ", mse)
    print("MAE: ", mae)

print("----------------------------------------------")
print("cost_arr:", cost_arr)
print("mse_arr:", mse_arr)
print("mae_arr:", mae_arr)
print("==============================")

y = scaler.inverse_transform(Y)
total_mse = mean_squared_error(y, predict_arr)
total_mae = mean_absolute_error(y, predict_arr)
print("total mse:", total_mse)
print("total mae:", total_mae)
np.save('predict_arr2.npy', predict_arr)
