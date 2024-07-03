import numpy as np
import pandas as pd
from functions import split, return_dictionaries
from movie_dataset_new import movies_new


df = pd.read_csv("kaggle\wiki_movie_plots_deduped.csv")
df = df.drop(columns=["Release Year", "Origin/Ethnicity", "Director",
                      "Cast", "Wiki Page"])
df = df[df.Genre != "unknown"]
df["Plot_Length"] = [len(plot) for plot in df["Plot"]]
df = df.drop_duplicates(subset=['Title'])
intersection = np.array([])
for movie in df["Title"].to_numpy():
    intersection = np.append(intersection, movie in movies_new.flatten())
all_movies = (df.Plot_Length > 5000).to_numpy()
intersection = all_movies + intersection
df["Intersection"] = intersection
df = df[df.Intersection >= 1]
counter = 0
total = 0
for movie in movies_new:
    movie = movie[0]
    if movie in df["Title"].to_numpy():
        counter += 1
        total += 1
    else:
        total += 1
kaggle_summaries = {}
kaggle_genres = {}
kaggle_movies = {}
genres_general = np.unique(pd.DataFrame(
    return_dictionaries(movies_new)[1]).to_numpy()[0])
genres_general = np.vectorize(str.lower)(genres_general)
new_genres = {}


def split_slash(input):
    if type(input) is str:
        return (input.split("/"))


def split_space(input):
    if type(input) is str:
        return (input.split(" "))


for r in range(np.shape(df)[0]):
# =============================================================================
#     title = df["Title"].to_numpy()[r]
#     kaggle_summaries[title] = df["Plot"].to_numpy()[r]
#     kaggle_movies[r] = [title]
#     kaggle_genres[r] = [genres[0], len(genres)]
# =============================================================================
    genres = split(df["Genre"].to_numpy()[r])
    genres = [g.strip() for g in genres]
    if "comedy" in genres:
        continue
    elif "drama" in genres:
        continue
    
    if genres[0] == "anime":
        title = df["Title"].to_numpy()[r]
        kaggle_summaries[title] = df["Plot"].to_numpy()[r]
        kaggle_movies[r] = [title]
        kaggle_genres[r] = [genres[0], len(genres)]
        new_genres[r] = ["animation", len(genres)]
    if genres[0] not in genres_general:
        genre = np.array(split_space(genres[0]))
        g_bool = np.array([], dtype=bool)
        for g in genre:
            g_bool = np.append(g_bool, g in genres_general)
        existing_genres = genre[g_bool]
        if len(existing_genres) > 0:
            title = df["Title"].to_numpy()[r]
            kaggle_summaries[title] = df["Plot"].to_numpy()[r]
            kaggle_movies[r] = [title]
            kaggle_genres[r] = [genres[0], len(genres)]
            choice = np.random.choice(existing_genres, p=np.repeat(
                1/len(existing_genres), len(existing_genres)))
            new_genres[r] = [str(choice), len(genres)]
    else:
        title = df["Title"].to_numpy()[r]
        kaggle_summaries[title] = df["Plot"].to_numpy()[r]
        kaggle_movies[r] = [title]
        kaggle_genres[r] = [genres[0], len(genres)]
        new_genres[r] = [genres[0], len(genres)]

