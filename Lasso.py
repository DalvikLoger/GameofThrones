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
from sklearn.metrics import mean_squared_error
#from interactions import show_log_regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,)

X_train_scaled = preprocessing.scale(X_train)

#Lasso
from sklearn.linear_model import Lasso

lasso_r = Lasso(alpha=1)

lasso_r.fit(X_train, y_train)

lasso_reg = Lasso(alpha=0.1)

lasso_reg.fit(X_train, y_train)

lasso_coef = lasso_reg.coef_

plt.plot(range(len(X.columns)), lasso_coef)
plt.xticks(range(len(X.columns)), X.columns.values, rotation=70);

print('score train :', lasso_reg.score(X_train, y_train))
print('score test :', lasso_reg.score(X_test, y_test))

lasso_pred_train = lasso_reg.predict(X_train)
lasso_pred_test = lasso_reg.predict(X_test)

print('mse train :', mean_squared_error(lasso_pred_train, y_train))
print('mse test :', mean_squared_error(lasso_pred_test, y_test))
