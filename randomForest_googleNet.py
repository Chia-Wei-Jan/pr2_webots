import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor

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

def randomForest(X_train, X_test, Y_train, Y_test):
    # create regressor object
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)

    # fit the regressor with data
    regressor.fit(X_train, Y_train)
    print("score:", regressor.score(X_train, Y_train))

    # Use train data to get RMSE
    Y_train_pred = regressor.predict(X_train)
    predict_train = scaler.inverse_transform(Y_train_pred)
    y_train_origin = scaler.inverse_transform(Y_train)

    rmse = mean_squared_error(y_train_origin, predict_train, squared=False)

    # predict
    Y_pred = regressor.predict(X_test)

    return Y_pred, rmse


def googleNet(X_train, X_test, Y_train, Y_test):
    inputs = keras.Input(shape=(11,))
    hA1 = layers.Dense(20, activation="relu")(inputs)
    hA1_1 = layers.Dense(20, activation="relu")(hA1)
    hB1_1 = layers.Dense(20, activation="relu")(hA1)
    hB1_2 = layers.Dense(20, activation="relu")(hB1_1)
    hC1_1 = layers.Dense(20, activation="relu")(hA1)
    hC1_2 = layers.Dense(20, activation="relu")(hC1_1)
    hD1_1 = layers.Dense(20, activation="relu")(hA1)
    hD1_2 = layers.Dense(20, activation="relu")(hD1_1)
    average1 = layers.Average()([hA1_1, hB1_2, hC1_2, hD1_2])
    hA2 = layers.Dense(20, activation="relu")(average1)
    hA2_1 = layers.Dense(20, activation="relu")(hA2)
    hB2_1 = layers.Dense(20, activation="relu")(hA2)
    hB2_2 = layers.Dense(20, activation="relu")(hB2_1)
    hC2_1 = layers.Dense(20, activation="relu")(hA2)
    hC2_2 = layers.Dense(20, activation="relu")(hC2_1)
    hD2_1 = layers.Dense(20, activation="relu")(hA2)
    hD2_2 = layers.Dense(20, activation="relu")(hD2_1)
    average2 = layers.Average()([hA2_1, hB2_2, hC2_2, hD2_2])
    hA3 = layers.Dense(20, activation="relu")(average2)
    hA3_1 = layers.Dense(20, activation="relu")(hA3)
    hB3_1 = layers.Dense(20, activation="relu")(hA3)
    hB3_2 = layers.Dense(20, activation="relu")(hB3_1)
    hC3_1 = layers.Dense(20, activation="relu")(hA3)
    hC3_2 = layers.Dense(20, activation="relu")(hC3_1)
    hD3_1 = layers.Dense(20, activation="relu")(hA3)
    hD3_2 = layers.Dense(20, activation="relu")(hD3_1)
    average3 = layers.Average()([hA3_1, hB3_2, hC3_2, hD3_2])
    hA4 = layers.Dense(20, activation="relu")(average3)
    hA4_1 = layers.Dense(20, activation="relu")(hA4)
    hB4_1 = layers.Dense(20, activation="relu")(hA4)
    hB4_2 = layers.Dense(20, activation="relu")(hB4_1)
    hC4_1 = layers.Dense(20, activation="relu")(hA4)
    hC4_2 = layers.Dense(20, activation="relu")(hC4_1)
    hD4_1 = layers.Dense(20, activation="relu")(hA4)
    hD4_2 = layers.Dense(20, activation="relu")(hD4_1)
    hE4_1 = layers.Dense(20, activation="relu")(hA4)
    hE4_2 = layers.Dense(20, activation="relu")(hE4_1)
    average4 = layers.Average()([hA4_1, hB4_2, hC4_2, hD4_2])
    outputs1 = layers.Dense(2, activation='sigmoid')(hE4_2)  # first output

    hA5 = layers.Dense(20, activation="relu")(average4)
    hA5_1 = layers.Dense(20, activation="relu")(hA5)
    hB5_1 = layers.Dense(20, activation="relu")(hA5)
    hB5_2 = layers.Dense(20, activation="relu")(hB5_1)
    hC5_1 = layers.Dense(20, activation="relu")(hA5)
    hC5_2 = layers.Dense(20, activation="relu")(hC5_1)
    hD5_1 = layers.Dense(20, activation="relu")(hA5)
    hD5_2 = layers.Dense(20, activation="relu")(hD5_1)
    average5 = layers.Average()([hA5_1, hB5_2, hC5_2, hD5_2])
    hA6 = layers.Dense(20, activation="relu")(average5)
    hA6_1 = layers.Dense(20, activation="relu")(hA6)
    hB6_1 = layers.Dense(20, activation="relu")(hA6)
    hB6_2 = layers.Dense(20, activation="relu")(hB6_1)
    hC6_1 = layers.Dense(20, activation="relu")(hA6)
    hC6_2 = layers.Dense(20, activation="relu")(hC6_1)
    hD6_1 = layers.Dense(20, activation="relu")(hA6)
    hD6_2 = layers.Dense(20, activation="relu")(hD6_1)
    average6 = layers.Average()([hA6_1, hB6_2, hC6_2, hD6_2])
    hA7 = layers.Dense(20, activation="relu")(average6)
    hA7_1 = layers.Dense(20, activation="relu")(hA7)
    hB7_1 = layers.Dense(20, activation="relu")(hA7)
    hB7_2 = layers.Dense(20, activation="relu")(hB7_1)
    hC7_1 = layers.Dense(20, activation="relu")(hA7)
    hC7_2 = layers.Dense(20, activation="relu")(hC7_1)
    hD7_1 = layers.Dense(20, activation="relu")(hA7)
    hD7_2 = layers.Dense(20, activation="relu")(hD7_1)
    hE7_1 = layers.Dense(20, activation="relu")(hA7)
    hE7_2 = layers.Dense(20, activation="relu")(hE7_1)
    average7 = layers.Average()([hA7_1, hB7_2, hC7_2, hD7_2])
    outputs2 = layers.Dense(2, activation='sigmoid')(hE7_2)  # second output

    hA8 = layers.Dense(20, activation="relu")(average7)
    hA8_1 = layers.Dense(20, activation="relu")(hA8)
    hB8_1 = layers.Dense(20, activation="relu")(hA8)
    hB8_2 = layers.Dense(20, activation="relu")(hB8_1)
    hC8_1 = layers.Dense(20, activation="relu")(hA8)
    hC8_2 = layers.Dense(20, activation="relu")(hC8_1)
    hD8_1 = layers.Dense(20, activation="relu")(hA8)
    hD8_2 = layers.Dense(20, activation="relu")(hD8_1)
    average8 = layers.Average()([hA8_1, hB8_2, hC8_2, hD8_2])
    hA9 = layers.Dense(20, activation="relu")(average8)
    hA9_1 = layers.Dense(20, activation="relu")(hA9)
    hB9_1 = layers.Dense(20, activation="relu")(hA9)
    hB9_2 = layers.Dense(20, activation="relu")(hB9_1)
    hC9_1 = layers.Dense(20, activation="relu")(hA9)
    hC9_2 = layers.Dense(20, activation="relu")(hC9_1)
    hD9_1 = layers.Dense(20, activation="relu")(hA9)
    hD9_2 = layers.Dense(20, activation="relu")(hD9_1)
    average9 = layers.Average()([hA9_1, hB9_2, hC9_2, hD9_2])
    hA10 = layers.Dense(20, activation="relu")(average9)
    outputs3 = layers.Dense(2, activation='sigmoid')(hA10)  # third output

    model = keras.Model(inputs=inputs, outputs=[outputs1, outputs2, outputs3])
    model.summary()
    model.compile(loss='mse', optimizer='sgd', metrics=['mse'])

    history = model.fit(X_train, [Y_train, Y_train, Y_train], epochs=10000, batch_size=8)

    # Use train data to get RMSE
    Y_pred_train = model.predict(X_train)
    predict_train = scaler.inverse_transform(Y_pred_train[2])
    y_train_origin = scaler.inverse_transform(Y_train)
    rmse = mean_squared_error(y_train_origin, predict_train, squared=False)

    # testing
    print("start testing")
    cost = model.evaluate(X_test, [Y_test, Y_test, Y_test])
    cost_arr.append(cost)
    print("test cost: {}".format(cost))

    Y_pred = model.predict(X_test)
    predict = scaler.inverse_transform(Y_pred[2])     # choose the deepest output

    return Y_pred[2], rmse


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    pred1, rmse1 = googleNet(X_train, X_test, Y_train, Y_test)
    pred2, rmse2 = randomForest(X_train, X_test, Y_train, Y_test)

    print("rmse1:", rmse1)
    print("rmse2:", rmse2)
    print("=======================")
    Y_pred = (pred1 * rmse1 + pred2 * rmse2) / (rmse1 + rmse2)

    predict = scaler.inverse_transform(Y_pred)
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
total_rmse = mean_squared_error(y, predict_arr, squared=False)
total_mae = mean_absolute_error(y, predict_arr)
print("total mse:", total_mse)
print("total rmse:", total_rmse)
print("total mae:", total_mae)
np.save('randomForest_googleNet.npy', predict_arr)
