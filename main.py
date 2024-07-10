import numpy as np
import pandas as pd
import os
from text_embedding_and_visualization import run_all

if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    
    datasets = ["data_movies\\final_data.csv"]
    models = [
        "BAAI/bge-m3",
        "intfloat/e5-large-v2",
        'whaleloops/phrase-bert',
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2"
    ]
    n = 10
    content_column = ["Description"]
    title_column = ["Name"]
    classify_language = ["Name"]

    results = run_all(
        datasets=datasets,
        models=models,
        n=n,
        create_graph=True,
        batch_size=32,
        content_column=content_column,
        title_column=title_column,
        classify_language=classify_language,
        duplicate_method='suffix',
        use_rag=False,
        rag_model="google/flan-t5-base",
        rag_embeddings_model="sentence-transformers/all-MiniLM-L6-v2",
        query_column="Description",
        force_new_embeddings=False,
        embeddings_only=True
    )

    print("Processing complete. Results stored in 'results' dictionary.")