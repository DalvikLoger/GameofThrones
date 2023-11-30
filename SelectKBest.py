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

#SelectKBest

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression

sk = SelectKBest(f_regression, k=3)

sk.fit(X=X, y=y)

X.columns[sk.get_support()]

sk_train = sk.transform(X_train)
sk_test = sk.transform(X_test)

sklr = LinearRegression()
sklr.fit(sk_train, y_train)

print(sklr.score(sk_train, y_train))
print(sklr.score(sk_test, y_test))