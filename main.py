import numpy as np
import pandas as pd
import os
from gephi_export import *
n = 5000
#gephi_export("data\stock_data.csv", 'whaleloops/phrase-bert', "Text", n)

datasets = ["data\\" + dataset for dataset in os.listdir("data")]
models = ['whaleloops/phrase-bert', "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"]
model_dict = run_all(datasets, models, n, graph=True)