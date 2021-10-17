import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

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

    # create regressor object
    svr = SVR()
    msvr = MultiOutputRegressor(svr)
    # fit the regressor with data
    msvr.fit(X_train, Y_train)
    print("score:", msvr.score(X_train, Y_train))

    # predict
    Y_pred = msvr.predict(X_test)
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
total_mae = mean_absolute_error(y, predict_arr)
print("total mse:", total_mse)
print("total mae:", total_mae)
