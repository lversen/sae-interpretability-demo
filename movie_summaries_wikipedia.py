import pandas as pd
import wikipediaapi as wp
import numpy as np
from functions import remove_first_sentence
movies = pd.read_csv("data\movies.txt", sep=" \n",
                     encoding='latin-1', engine='python', header=None)
movies = movies.to_numpy()

ww = wp.Wikipedia("Project paper", "en")


movies_new = open("movies_new.txt", "w")
counter = 0
for movie in movies:
    movie = movie[0]
    summary = ww.page(movie + "_(movie)").summary
    if (len(summary) > 500 and len(remove_first_sentence(summary)) > 0):
        try:
            try:
                movie_summary = open("summaries\summary_" + movie + ".txt", "w")
                movie_summary.write(summary)
                movie_summary.close()
                movies_new.write(movie + " \n")
                counter += 1
            except OSError:
                print(movie + " was skipped")
        except UnicodeEncodeError:
            print(movie + " was skipped")

movies_new.close()