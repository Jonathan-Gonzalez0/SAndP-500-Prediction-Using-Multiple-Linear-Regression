# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:43:32 2025

@author: Jonathan Gonzalez

Machine Learning Regression Masterclass in Python 
By: Dr. Ryan Ahmed 
Platform: Udemy
Type: Compilation of videos

This Python script analyzes the relationship between stock market 
prices (S&P 500 Price), employment rates, and interest rates using 
multiple linear regression.

Last Updated: 1/14/2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SAndP500Data = pd.read_csv("S&P500_Stock_Data.csv")

print(SAndP500Data.describe())

print("\n")
print(SAndP500Data.info())

plt.close("all")
sns.jointplot(x="Employment", y = "S&P 500 Price", data = SAndP500Data)
sns.jointplot(x="Interest Rates", y = "S&P 500 Price", data = SAndP500Data)

sns.pairplot(SAndP500Data)

y = SAndP500Data["S&P 500 Price"]

x = SAndP500Data[["Interest Rates", "Employment"]]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size =0.2,random_state = 1)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression(fit_intercept = True)

regressor.fit(x_train,y_train)

print("Linear Model Coefficients (m)", regressor.coef_)
print("Linear Model Coefficients (b)", regressor.intercept_)

y_predict = regressor.predict(x_test)

plt.figure()
plt.scatter(y_test, y_predict, color = "r")
plt.ylabel("Model Predictions")
plt.xlabel("True Values (Ground Truth)")
plt.title("Multiplr Linear Regression Predictions")

k = x_test.shape[1]

n = len(x_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt

RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)) , ".3f" ))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test,y_predict)
r2 = r2_score(y_test,y_predict)
adj_r2 = 1- (1-r2)* (n-1)/(n-k-1)
MAPE = np.mean( np.abs( (y_test-y_predict) / y_test ))*100
print("\n")

print("RMSE = ", RMSE, "\nMSE =", MSE, "\nMAE =", MAE, "\nR2 =", r2, "\nAdjusted R2 =", adj_r2, "\nMAPE =", MAPE)

from mpl_toolkits.mplot3d import Axes3D

x_surf, y_surf = np.meshgrid(np.linspace( SAndP500Data["Interest Rates"].min(), SAndP500Data["Interest Rates"].max(), 100), np.linspace(SAndP500Data["Employment"].min(), SAndP500Data["Employment"].max(), 100))
                                                                                                                                        
onlyX = pd.DataFrame( {"Interest Rates": x_surf.ravel(), "Employment": y_surf.ravel() } )                                                                                                                       

fittedY = regressor.predict(onlyX)

fittedY = fittedY.reshape(x_surf.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter( SAndP500Data["Interest Rates"], SAndP500Data["Employment"], SAndP500Data["S&P 500 Price"], c = "blue", marker = "x" )
ax.plot_surface(x_surf, y_surf, fittedY, color = "red", alpha = 0.3)
ax.set_xlabel("Interest Rates")
ax.set_ylabel("Unemployment Rates")
ax.set_zlabel("Stock Index Price")
ax.view_init(20,30)
plt.show()

coeff_df = pd.DataFrame(regressor.coef_, x.columns, columns = ["Coefficient"])

print(coeff_df)