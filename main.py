import numpy as np
import pandas as pd
import os
from gephi_export_stocks import *
n = 5000
#gephi_export("data\stock_data.csv", 'whaleloops/phrase-bert', "Text", n)
folder = "data"
datasets = [folder + "\\" + dataset for dataset in os.listdir(folder)]
models = ['whaleloops/phrase-bert', "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"]
model_dict = run_all(datasets, models, n, graph=True)