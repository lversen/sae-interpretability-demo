import wikipediaapi as wp
import numpy as np
from movie_dataset import movie_dict

ww = wp.Wikipedia("Project paper", "en")

df = movie_dict["Movie"]

movies = open("movies.txt", "w")
counter = 0
n_movies = 1000
for movie in df:
    try:
        summary = ww.page(movie + "_(movie)").summary[:1]
    except KeyError:
        df = df.drop(df[df == movie].index)
    if len(summary) > 0:
        movies.write(movie + " \n")
        counter += 1
    if counter == n_movies:
        break
movies.close()

