import numpy as np
import pandas as pd
import os
from gephi_export_stocks import *
n = 10_000
#models = ["BAAI/bge-m3", "intfloat/e5-large-v2", "Alibaba-NLP/gte-large-en-v1.5", 'whaleloops/phrase-bert', "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"]
models = ["Alibaba-NLP/gte-large-en-v1.5"]
run_all(["data_movies\\final_data.csv"], models, n, graph=True, batch_size=16, content_column=["Description"], title_column=["Name"], language_classes=True)

#datasets = ["data\\" + dataset for dataset in os.listdir("data")]
#datasets = ["data\\Project6500.csv"]
#model_dict = run_all(datasets, models, n, graph=True, batch_size=32, iterations=1)


# MAKE RAG
# TRY TRANSLATING MOVIE DESCRIPTIONS TO ENGLISH
#https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/
#USE BGE MODEL
# TRY OPENAI ADA EMBEDDINGS from langchain.embeddings import OpenAIEmbeddings

#94 percent of languages were english all of the sudden at n=10_000