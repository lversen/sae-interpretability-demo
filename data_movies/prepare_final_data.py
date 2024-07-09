import numpy as np
import pandas as pd

df = pd.read_csv("data_movies\\final_data.csv")
df = df.drop(["url", "ReviewBody", "PosterLink", "ReviewAurthor", "ReviewDate", "duration", "BestRating", "WorstRating", "id"], axis="columns")
df = df.dropna()
df = df[df.RatingCount>0]
df = df[df.RatingValue>0]
df = df.to_csv("data_movies\\final_data.csv", index=False)