import numpy as np
import pandas as pd

df = pd.read_csv("data_movies\\final_data.csv", encoding = "ISO-8859-1")
""" df = df.drop(["url", "ReviewBody", "PosterLink", "ReviewAurthor", "ReviewDate", "duration", "BestRating", "WorstRating", "id"], axis="columns")
df = df.dropna()
df = df[df.RatingCount>0]
df = df[df.RatingValue>0]
 """
""" l = list(df.columns)
l.remove("Unnamed: 0.1")
df = df[l] """
df['Description'] = df['Description'].map(lambda x: x.replace("Ã", ""))
df = df.to_csv("data_movies\\final_data.csv", index=False)