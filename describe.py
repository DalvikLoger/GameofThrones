#Game of Throne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.image import imread
from bokeh.plotting import figure, show, output_notebook
import seaborn as sns
df = pd.read_csv('Game_of_Thrones.csv',header=0 )

IMDB_Rating_mean = df['IMDb Rating'].mean()

Viewers_mean = df['U.S. Viewers (Millions)'].mean()

Episode_viewers_max = df[df['U.S. Viewers (Millions)'] == 13.61]

Best_Episode = df.sort_values(by = ['IMDb Rating','Metacritic Ratings'], ascending = False).head(5)

Worst_Episode = df.sort_values(by = ['IMDb Rating','Metacritic Ratings'], ascending = True).head(5)

print('The mean critics rating is', IMDB_Rating_mean, '.')

print('The mean US viewers of all episode is ', Viewers_mean, '.')

print(" There is more information about the episode which has the most viewers :", Episode_viewers_max)

print("There are the episodes with the best critics and viewers : ", Best_Episode)

print("There are the episodes with the worst critics and viewers : ", Worst_Episode)