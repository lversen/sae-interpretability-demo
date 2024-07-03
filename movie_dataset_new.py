import pandas as pd
import numpy as np
from functions import *
movies_new = pd.read_csv("movies_new.txt", sep=" \n",
                         encoding='latin-1', engine='python', header=None)
movies_new = movies_new.to_numpy()

sum_short = {}
sum_long = {}

for movie in movies_new:
    try:
        movie = movie[0]
        summary = pd.read_csv("wikipedia\summaries\summary_" + movie + ".txt", sep=" \n",
                            encoding='latin-1', engine='python', header=None)
        summary[0][0] = summary[0][0].replace(movie, "")
        s = ""
        for i, S in enumerate(summary[0]):
            s += S + "\n"
            if i == 0:
                sum_short[movie] = s
        sum_long[movie] = s
    except FileNotFoundError:
        print(movie + " was not found")