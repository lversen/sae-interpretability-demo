import numpy as np
import pandas as pd
import os
from gephi_export_stocks import *
n = 50 # length of tweet dataset

#models = ["Alibaba-NLP/gte-large-en-v1.5", 'whaleloops/phrase-bert', "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"]
models = ["Alibaba-NLP/gte-large-en-v1.5"]
#run_all(["data_movies\\final_data.csv"], models, n, graph=True, batch_size=16)

#datasets = ["data\\" + dataset for dataset in os.listdir("data")]
datasets = ["data\\Project6500.csv"]
model_dict = run_all(datasets, models, n, graph=True)