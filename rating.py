#Game of Throne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.image import imread
from bokeh.plotting import figure, show, output_notebook
import seaborn as sns
df = pd.read_csv('Game_of_Thrones.csv',header=0 )

x = df['No. of Episode (Overall)']      
y1 = df['U.S. Viewers (Millions)']     
y2 = df['IMDb Rating']   

fig = plt.figure(figsize=(30,20))

plt.subplot(221)
plt.plot(x, y1, label = 'Nbr Viewers')
plt.legend()

plt.subplot(223)
plt.plot(x, y2, c = 'm', label = "Rating",)
plt.legend()
