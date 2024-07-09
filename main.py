import numpy as np
import pandas as pd
import os
#from gephi_export_stocks import *
from test import run_all
#n = 10_000
#models = ["BAAI/bge-m3", "intfloat/e5-large-v2", "Alibaba-NLP/gte-large-en-v1.5", 'whaleloops/phrase-bert', "paraphrase-MiniLM-L6-v2", "all-mpnet-base-v2"]
#models = ["BAAI/bge-m3"]
#run_all(["data_movies\\final_data.csv"], models, n, graph=False, batch_size=16,
        #content_column=["Description"], title_column=["Name"], classify_language=["Name"])

#datasets = ["data\\" + dataset for dataset in os.listdir("data")]
#datasets = ["data\\Project6500.csv"]
#model_dict = run_all(datasets, models, n, graph=True, batch_size=32, iterations=1)


# MAKE RAG
#https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/
#USE BGE MODEL
# TRY OPENAI ADA EMBEDDINGS from langchain.embeddings import OpenAIEmbeddings





if __name__ == "__main__":
    datasets = ["data_movies\\final_data.csv"]
    models = models = [
        "BAAI/bge-m3",
        #"intfloat/e5-large-v2",
        #"Alibaba-NLP/gte-large-en-v1.5",
        #'whaleloops/phrase-bert',
        #"paraphrase-MiniLM-L6-v2",
        #"all-mpnet-base-v2"
        ]
    n = 1000  # number of samples to process
    content_column = ["Description"]
    title_column = ["Name"]
    classify_language = ["Name"]


    results = run_all(
        datasets,
        models,
        n,
        graph=True,
        batch_size=32,
        content_column=content_column,
        title_column=title_column,
        classify_language=classify_language,
        duplicate_method='suffix',
        use_rag=True,
        rag_model="google/flan-t5-base",
        rag_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        query_column="query_column",
        force_new_embeddings=False
    )

    print("Processing complete. Results stored in 'results' dictionary.")

if __name__ == "__main__":
    datasets = ["data_movies\\final_data.csv"]
    models = models = [
        "BAAI/bge-m3",
        #"intfloat/e5-large-v2",
        #"Alibaba-NLP/gte-large-en-v1.5",
        #'whaleloops/phrase-bert',
        #"paraphrase-MiniLM-L6-v2",
        #"all-mpnet-base-v2"
        ]
    n = 1000  # number of samples to process
    content_column = ["Description"]
    title_column = ["Name"]
    classify_language = ["Name"]
    
    results = run_all(
        datasets,
        models,
        n,
        graph=True,
        batch_size=32,
        content_column=content_column,
        title_column=title_column,
        classify_language=classify_language,
        duplicate_method='suffix',
        use_rag=True,
        rag_model="google/flan-t5-base",
        rag_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        query_column="query_column",
        force_new_embeddings=False
    )

    print("Processing complete. Results stored in 'results' dictionary.")