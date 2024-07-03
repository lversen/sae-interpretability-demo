import numpy as np
import pandas as pd
import os
from gephi_export_stocks import *
n = 5000

datasets = ["data\\" + dataset for dataset in os.listdir("data")]
models = ["Alibaba-NLP/gte-large-en-v1.5", 'whaleloops/phrase-bert', "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"]
model_dict = run_all(datasets, models, n, graph=True)