import numpy as np
import pandas as pd
import os
from gephi_export_stocks import *
n = 10
models = ["BAAI/bge-m3", "intfloat/e5-large-v2", "Alibaba-NLP/gte-large-en-v1.5", 'whaleloops/phrase-bert', "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"]

run_all(["data_movies\\final_data.csv"], models, n, graph=True, batch_size=16, content_column=["Description"], title_column=["Name"])

#datasets = ["data\\" + dataset for dataset in os.listdir("data")]
#datasets = ["data\\Project6500.csv"]
#model_dict = run_all(datasets, models, n, graph=True, batch_size=32, iterations=1)