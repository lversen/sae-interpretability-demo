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


if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    datasets = ["data_movies\\final_data.csv"]
    models = [
        "BAAI/bge-m3",
        "intfloat/e5-large-v2",
        'whaleloops/phrase-bert',
        "paraphrase-MiniLM-L6-v2",
        "all-mpnet-base-v2"
        ]
    n = 10
    content_column = ["Description"]
    title_column = ["Name"]
    classify_language = ["Name"]

    results = run_all(
        datasets=datasets,
        models=models,
        n=n,
        graph=True,
        batch_size=32,
        content_column=content_column,
        title_column=title_column,
        classify_language=classify_language,
        duplicate_method='suffix',
        use_rag=False,
        rag_model="google/flan-t5-base",
        rag_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        query_column="Description",
        force_new_embeddings=False
    )

    print("Processing complete. Results stored in 'results' dictionary.")