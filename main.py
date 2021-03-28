import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow import keras
from keras.models import Sequential
from keras.layers.core import Dense,Dropout
from keras.layers import LSTM
from keras import activations,backend
from scipy import spatial


def train_model(model, model_name):
    accuracy = {}
    rmse = {}
    explained_variance = {}
    max_error = {}
    MAE = {}
    COS = {}

    print(model_name)
    model.fit(X_train, y_train.ravel())
    pred = model.predict(X_test)

    #reverse data back to normal form
    pred = scaler.inverse_transform(pred)

    acc = metrics.r2_score(y_test, pred) * 100
    print('R2_Score', acc)
    accuracy[model_name] = acc

    met = np.sqrt(metrics.mean_squared_error(y_test, pred))
    print('MSE : ', met)
    rmse[model_name] = met

    var = (metrics.explained_variance_score(y_test, pred))
    print('Explained_Variance : ', var)
    explained_variance[model_name] = var

    error = (metrics.max_error(y_test, pred))
    print('Max_Error : ', error)
    max_error[model_name] = error

    err = metrics.mean_absolute_error(y_test, pred)
    print("Mean Absolute Error", err)
    MAE[model_name] = err

    cos = 1 - spatial.distance.cosine(y_test, pred)
    print("Cosine Similarity", cos)
    COS[model_name] = cos

    plt.close()
    plt.scatter(range(y_test.shape[0]),y_test, label="Actual Value")
    plt.scatter(range(pred.shape[0]), pred, label='Prediction')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(model_name)
    plt.show()

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)



if __name__ == '__main__':
    #simplefilter("ignore", category=ConvergenceWarning)

    # Fortonoume ta dedomena
    data = pd.read_csv("features.csv")

    #print(data.iloc[:,5:12].head())
    x = data.info()
    print("\n")
    x = data.describe(include='all')
    print(x)

    # Check for missing values
    missing = 100*(data.isna().sum())/len(data)
    print("Missing values percentage before preprocessing:")
    print(missing)


    # Gemizoume tis elipis times tou Item_Weight me tin mesi timi
    data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)

    # Antistoixizoume tis times se arithmous kai simplirwnoume ta kena me tin mesi timi
    data['Outlet_Size'] = data['Outlet_Size'].map({'Small': 1, 'Medium': 2, 'High': 3})
    #print("The median value : ", data['Outlet_Size'].median())
    data['Outlet_Size'] = data['Outlet_Size'].fillna(data['Outlet_Size'].median())

    # print(data.groupby(['Outlet_Location_Type', 'Outlet_Type'])['Outlet_Size'].value_counts())
    # data.Outlet_Size = data.apply(func, axis=1)

    missing = 100 * (data.isna().sum()) / len(data)
    print("Missing values percentage after preprocessing:")
    print(missing)

    #Mapping timws se arithmous gia na mporesoume na xrisimopoihsoume
    data['Outlet_Type'] = data['Outlet_Type'].map({'Grocery Store': 1, 'Supermarket Type1': 2, 'Supermarket Type2': 3, 'Supermarket Type3': 4})
    data['Outlet_Location_Type'] = data['Outlet_Location_Type'].map({'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3})

    # Diorthwnoume tis times stin stili Item_Fat_Content
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('LF', 'Low Fat')
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('low fat', 'Low Fat')
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace('reg', 'Regular')


    # Dimiourgoume ta dianusmata
    # X = data.drop(["Item_Outlet_Sales"], axis=1).values.reshape((-1, 1))
    # Y = data["Item_Outlet_Sales"].values.reshape((-1, 1))
    # Y = [float(x) for x in Y]
    X = data.select_dtypes(include=np.number).drop(["Item_Outlet_Sales"], axis=1)
    y = data["Item_Outlet_Sales"]

    # Use only one feature
    # X = data["Item_MRP"]

    # Kanoume ena arxiko plot twn dianismatwn
    # plt.figure(figsize=(16, 8))
    # plt.scatter(X, y, c='black')
    # plt.xlabel("Item_MRP")
    # plt.ylabel("SALES")
    # plt.show()


    # Coolerelation Matrix
    corr = data.select_dtypes(include=[np.number]).corr()
    print(sns.heatmap(corr, annot=True, vmax=8, square=True))
    plt.show()

    X = X.astype(float)
    y = y.astype(float)


    # normalize data before feeding into LSTM
    scaler = StandardScaler()
    size = X.shape
    X = X.to_numpy().reshape(size[0],size[1])
    size = y.shape
    y = y.to_numpy().reshape(size[0],1)
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.3, random_state=1)


    # Model Training
    reg = LinearRegression()
    dtr = DecisionTreeRegressor(max_depth=15, min_samples_leaf=100)
    rfc = ensemble.RandomForestRegressor(n_estimators=400, bootstrap=True, min_samples_leaf=55)
    gbr = GradientBoostingRegressor()

    print(pd.DataFrame(X).describe())

    y_test = scaler.inverse_transform(y_test)
    print("\n")
    train_model(reg, "Linear Regression")
    print("-" * 30)
    train_model(dtr, "Decision Tree")
    print("-" * 30)
    train_model(rfc, "Random Forest")
    print("-" * 30)
    train_model(gbr, "Gradient Boosting Regression")

    #LSTM
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    print(device_lib.list_local_devices())
    # design network LSTM
    model = Sequential()
    model.add(LSTM(50, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]),return_sequences=True,
                   activation=activations.tanh, stateful=True))
    # model.add(Dropout(0.1))
    model.add(LSTM(32, return_sequences=True, activation=activations.tanh))
    # model.add(Dropout(0.1))
    model.add(Dense(units=10, activation=activations.relu))
    #model.add(Dropout(0.1))
    model.add(Dense(units=1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(X_train, y_train, epochs=25, batch_size=1, verbose=1)
    predictions = model.predict(X_test, batch_size=1)
    predictions = predictions.reshape(predictions.shape[0], 1)
    predictions = scaler.inverse_transform(predictions)
    acc = metrics.r2_score(y_test, predictions) * 100
    print("-" * 30)
    print("LSTM")
    print('R2_Score', acc)
    met = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    print('MSE : ', met)
    var = (metrics.explained_variance_score(y_test, predictions))
    print('Explained_Variance : ', var)
    error = (metrics.max_error(y_test, predictions))
    print('Max_Error : ', error)
    err = metrics.mean_absolute_error(y_test, predictions)
    print("Mean Absolute Error", err)
    print("Cosine Similarity", 1-spatial.distance.cosine(y_test,predictions))
    plt.clf()
    plt.scatter(range(y_test.shape[0]), y_test)
    plt.scatter(range(predictions.shape[0]), predictions)
    plt.title("LSTM")
    plt.show()


