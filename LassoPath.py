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

#Lasso Path
from sklearn.linear_model import lasso_path

mes_alphas = (0.001, 0.01, 0.02, 0.025, 0.05, 0.1, 0.25, 0.5, 0.8, 1.0)

alpha_path, coefs_lasso, _ = lasso_path(X_train, y_train, alphas=mes_alphas)

print(coefs_lasso.shape)

plt.figure(figsize=(10, 7))

for i in range(coefs_lasso.shape[0]):
    plt.plot(alpha_path, coefs_lasso[i,:], '--')

plt.xlabel('Alpha')
plt.ylabel('Coefficients')
plt.title('Lasso path');