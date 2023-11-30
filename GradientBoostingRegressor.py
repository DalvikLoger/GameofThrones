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


from sklearn.ensemble import GradientBoostingRegressor

# Ces arguments ont été choisis pour surapprendre le plus possible
# Ne pas les utiliser en pratique
gbr = GradientBoostingRegressor(n_estimators = 1000,
                                max_depth = 10000,
                                max_features = 0.25,
                                validation_fraction = 0)
gbr.fit(X_train, y_train)

y_pred_train_gbr = gbr.predict(X_train)

y_pred_test_gbr = gbr.predict(X_test)

from sklearn.metrics import mean_absolute_error

MAE_test = mean_absolute_error(y_test, y_pred_test_gbr)
MAE_train = mean_absolute_error(y_train, y_pred_train_gbr)

print("MAE train gbr:", MAE_train)
print("MAE test gbr:", MAE_test)

mean_price_gbr = df['IMDb Rating'].mean()
print("Relative error", MAE_test / mean_price_gbr)