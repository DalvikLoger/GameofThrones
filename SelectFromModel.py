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

#SelectFromModel

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

sfm = SelectFromModel(lr)

scaler = preprocessing.StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)

sfm_train = sfm.fit_transform(X_train_scaled, y_train)

sfm_test = sfm.transform(X_test_scaled)

X.columns[sfm.get_support()]

sfmlr = LinearRegression()
sfmlr.fit(sfm_train, y_train)

print(sfmlr.score(sfm_train, y_train))
print(sfmlr.score(sfm_test, y_test))
