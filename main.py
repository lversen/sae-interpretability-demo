from typing import List, Dict, Any
import pandas as pd
from feature_extraction_with_store import *
from gephi import *
from language_classification import *
from data_init import *
from sample_handler import *


def run_all(
    datasets: List[str],
    models: List[str],
    n: int,
    feature_column: List[str],
    label_column: List[str],
    create_graph: bool = True,
    force_new_embeddings: bool = False,
    classify_language = []
):
    
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")        
        # Load full dataset
        full_df = pd.read_csv(dataset)
        print(f"Full dataset shape: {full_df.shape}")
        for model in models:
            print(f"Processing model: {model}")
            
            # Get consistent samples
            df, indices = get_consistent_samples(full_df, n, dataset, model)
            print(f"Sample shape: {df.shape}")
            
            # Feature extraction
            feature_extract = feature_extraction_with_store(
                df, full_df, model, n, dataset, feature_column[i], 
                force_new_embeddings=force_new_embeddings
            )
            if len(classify_language) != 0:
                indices = np.array(indices, dtype=np.int32)
                language_classifier(df, indices, classify_language, dataset)
            if create_graph:
                # Gephi export
                mapping, attributes = node_attributes(df, label_column[i], model)
                print(f"Exporting Gephi graph for {dataset} with model {model}")
                gephi_export(feature_extract, dataset, model, mapping, attributes)


if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    

    datasets = ["data\\raw_analyst_ratings.csv"]
    feature_column = ["headline"]
    label_column = ["headline"]
    models = [
        "BAAI/bge-m3",
        #"intfloat/e5-large-v2",
        #'whaleloops/phrase-bert',
        #"sentence-transformers/paraphrase-MiniLM-L6-v2",
        #"sentence-transformers/all-mpnet-base-v2"
    ]
    n = 5000

    results = run_all(
        datasets=datasets,
        models=models,
        n=n,
        feature_column=feature_column,
        label_column=label_column,
        create_graph=True,
    )