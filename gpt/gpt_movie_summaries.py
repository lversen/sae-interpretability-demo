import numpy as np
import pandas as pd
import os
from functions import *
summaries_gpt = {}

movies_new = pd.read_csv("wikipedia\movies_new.txt", sep=" \n",
                         encoding='latin-1', engine='python', header=None)
movies_new = movies_new.to_numpy()
for movie in movies_new:
    movie = movie[0]
    if os.path.isfile("gpt\summaries\summarygpt_" + movie + ".txt"):
        summary = open("gpt\summaries\summarygpt_" + movie + ".txt", "r")
        summary = summary.read()
        summary = summary.replace(
            "After this input give a synopsis of this movie: " + movie, "")
        summary = summary.replace("movie", "")
        summaries_gpt[movie] = summary
    else:
        movies_new = np.delete(movies_new, np.where(movies_new == movie)[0])

movies_new_gpt = np.reshape(np.unique(movies_new),
                            (len(np.unique(movies_new)), 1))
