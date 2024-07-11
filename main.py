from typing import List, Dict, Any
import pandas as pd
from visualize_multi_category_graph import *
from feature_extraction_with_store import *
from classify_with_networkx import *
from gephi import *
from language_classification import *
from data_init import *
from sample_handler import *


def run_all(
    datasets: List[str],
    models: List[str],
    n: int,
    content_column: List[str],
    id_column: List[str],
    category_column: List[str],
    n_neighbors: int = 5,
    max_categories: int = 10,
    batch_size: int = 32,
    create_graph: bool = True,
    use_networkx_classification: bool = True,
    force_new_embeddings: bool = False,
    embeddings_only: bool = False
) -> Dict[str, Dict[str, Any]]:
    model_dict = {}
    
    for i, dataset in enumerate(datasets):
        print(f"Processing dataset: {dataset}")
        
        # Try different encodings
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                full_df = pd.read_csv(dataset, encoding=encoding)
                print(f"Successfully read the file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                print(f"Failed to read with {encoding} encoding")
        else:
            print(f"Unable to read the file with any of the attempted encodings")
            continue  # Skip to the next dataset if we can't read this one

        print(f"Full dataset shape: {full_df.shape}")
        
        # Load full dataset
        full_df = pd.read_csv(dataset)
        print(f"Full dataset shape: {full_df.shape}")
        
        for model in models:
            print(f"Processing model: {model}")
            
            # Get consistent samples
            df, selected_indices = get_consistent_samples(full_df, n, dataset, model)
            print(f"Sample shape: {df.shape}")
            print(f"Number of selected indices: {len(selected_indices)}")
            
            # Feature extraction
            feature_extract, vectorstore = feature_extraction_with_store(
                df, model, batch_size, n, dataset, content_column[i], 
                force_new_embeddings=force_new_embeddings,
                embeddings_only=embeddings_only
            )
            
            if model not in model_dict:
                model_dict[model] = {}
            if dataset not in model_dict[model]:
                model_dict[model][dataset] = {}
            
            model_dict[model][dataset]['feature_extract'] = feature_extract
            model_dict[model][dataset]['vectorstore'] = vectorstore
            
            if use_networkx_classification:
                print(f"Performing NetworkX classification for {dataset} with model {model}")
                df, graph = classify_with_networkx(
                    feature_extract, df, id_column[i], 
                    category_column[i], vectorstore if not embeddings_only else None, 
                    n_neighbors=n_neighbors
                )
                model_dict[model][dataset]['networkx_graph'] = graph
                model_dict[model][dataset]['classified_df'] = df
                
                if create_graph:
                    # Visualize the multi-category graph
                    output_file = f"multi_category_graph_{dataset.replace('/', '_')}_{model.replace('/', '_')}_n{n}.png"
                    visualize_multi_category_graph(graph, df, id_column[i], output_file, max_categories=max_categories)
                    print(f"Multi-category graph saved as {output_file}")
            
            # Gephi export
            print(f"Exporting Gephi graph for {dataset} with model {model}")
            mapping, attributes = node_attributes(df, id_column[i])
            gephi_export(feature_extract, dataset, model, mapping, attributes)
    
    return model_dict


if __name__ == "__main__":
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    
    datasets = ["data\Project6500.csv"]
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
        use_networkx_classification=False,
        create_graph=False,
        embeddings_only=True
    )

    print("Processing complete. Results stored in 'results' dictionary.")