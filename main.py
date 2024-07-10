import numpy as np
import pandas as pd
import os
from text_embedding_and_visualization import run_all

if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    
    datasets = ["data_movies\\final_data.csv"]
    models = [
        "BAAI/bge-m3",
        #"intfloat/e5-large-v2",
        #'whaleloops/phrase-bert',
        #"sentence-transformers/paraphrase-MiniLM-L6-v2",
        #"sentence-transformers/all-mpnet-base-v2"
    ]
    n = 10_000
    content_column = ["Description"]
    id_column = ["Name"]
    category_column = ["Genres"]
    classify_language = ["Name"]

    results = run_all(
        datasets=datasets,
        models=models,
        n=n,
        content_column=content_column,
        id_column=id_column,
        category_column=category_column,
        n_neighbors=10,
        max_categories=5,
        use_networkx_classification=True,
        create_graph=True,
        force_new_embeddings=False,
        embeddings_only=False
    )

    print("Processing complete. Results stored in 'results' dictionary.")