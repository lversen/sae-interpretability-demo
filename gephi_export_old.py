from functions import *
from movie_dataset_new import movies_new, sum_short, sum_long
from kaggle_dataset import kaggle_summaries, kaggle_movies, new_genres
from sentence_transformers import SentenceTransformer
wiki_dicts = return_dictionaries(movies_new)

# =============================================================================
# short = gephi_export(sum_short, wiki_dicts[0], wiki_dicts[1], "test_short.gexf")
# =============================================================================
long = gephi_export(sum_long, wiki_dicts[0], wiki_dicts[1], "test_long.gexf")
#kaggle = gephi_export(kaggle_summaries, kaggle_movies, new_genres, "test_kaggle.gexf")
# =============================================================================
# 
# model = SentenceTransformer('whaleloops/phrase-bert')
# 
# datasets = add_dataset(summaries_gpt, model)
# datasets = add_dataset(sum_short, model)
# datasets = add_dataset(sum_long, model)
# datasets = add_dataset(kaggle_summaries, model)
# dist = dist_matrix(datasets)
# 
# =============================================================================
 