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

from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X, y)

print(slr.intercept_)
print(slr.coef_)

print(cross_validate(slr, X, y, return_train_score=True, cv=4))

print(cross_validate(slr, X, y, return_train_score=True, cv=4)['test_score'].mean()
)
pred_prix = slr.predict(X)
residus = pred_prix - y
print('Residus :',residus.describe())

from sklearn.feature_selection import f_regression

print('F-statistique :', f_regression(X, y)[0], 'p-value :', f_regression(X, y)[1])

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets)**2).mean())

print('rmse :',rmse(pred_prix, y))
