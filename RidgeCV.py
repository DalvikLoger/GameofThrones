#Game of Throne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.image import imread
from bokeh.plotting import figure, show, output_notebook
import seaborn as sns
df = pd.read_csv('Game_of_Thrones.csv',header=0 )

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier
A = df['U.S. Viewers (Millions)']
B = df['Metacritic Ratings']
C = df['Rotten Tomatoes Rating (Percentage)']
X = pd.concat([A,B,C], axis=1)
y = df['IMDb Rating']

from sklearn.model_selection import train_test_split
from sklearn import linear_model, preprocessing
#from interactions import show_log_regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,)

X_train_scaled = preprocessing.scale(X_train)

#RidgeCV
from sklearn.linear_model import RidgeCV

ridge_reg = RidgeCV(alphas= (0.001, 0.01, 0.1, 0.3, 0.7, 1, 10, 50, 100))
ridge_reg.fit(X_train, y_train)

print('alpha sélectionné par c-v :', ridge_reg.alpha_)
print('score train :', ridge_reg.score(X_train, y_train))
print('score test :', ridge_reg.score(X_test, y_test))

from sklearn.metrics import mean_squared_error
ridge_pred_train = ridge_reg.predict(X_train)
ridge_pred_test = ridge_reg.predict(X_test)

print('mse train :', mean_squared_error(ridge_pred_train, y_train))
print('mse test :', mean_squared_error(ridge_pred_test, y_test))
