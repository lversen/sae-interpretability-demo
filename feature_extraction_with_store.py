import os
import numpy as np
import pandas as pd
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def feature_extraction_with_store(df: pd.DataFrame, model: str, batch_size: int, n: int, dataset_name: str, content_column: str, force_new_embeddings: bool = False, embeddings_only: bool = False):
    base_dir = os.path.join("vectorstores_and_embeddings", dataset_name, model.replace('/', '_'))
    os.makedirs(base_dir, exist_ok=True)
    
    embeddings_path = os.path.join(base_dir, "embeddings.npy")
    vectorstore_path = os.path.join(base_dir, "vectorstore")
    index_path = os.path.join(base_dir, "index_max.pkl")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=model)
    
    # Load or create index
    if os.path.exists(index_path):
        with open(index_path, 'rb') as f:
            all_indices = pickle.load(f)
    else:
        all_indices = np.random.permutation(df.index)
        with open(index_path, 'wb') as f:
            pickle.dump(all_indices, f)
    
    selected_indices = all_indices[:n]
    
    # Load existing embeddings if available
    if os.path.exists(embeddings_path) and not force_new_embeddings:
        all_embeddings = np.load(embeddings_path)
        existing_n = len(all_embeddings)
    else:
        all_embeddings = np.array([])
        existing_n = 0
    
    # Determine which embeddings need to be computed
    new_indices = [idx for idx in selected_indices if idx >= existing_n]
    
    if len(new_indices) > 0 or force_new_embeddings:
        print(f"Computing embeddings for {len(new_indices)} new samples")
        new_texts = df.loc[new_indices, content_column].tolist()
        new_embeddings = embeddings.embed_documents(new_texts)
        
        if len(all_embeddings) > 0 and not force_new_embeddings:
            all_embeddings = np.vstack([all_embeddings, new_embeddings])
        else:
            all_embeddings = np.array(new_embeddings)
        
        np.save(embeddings_path, all_embeddings)
    
    # Select only the required embeddings
    mask = np.isin(all_indices[:len(all_embeddings)], selected_indices)
    feature_extract = all_embeddings[mask]
    
    print(f"Embedding statistics:")
    print(f"  Shape: {feature_extract.shape}")
    print(f"  Mean: {np.mean(feature_extract):.4f}")
    print(f"  Std: {np.std(feature_extract):.4f}")
    print(f"  Min: {np.min(feature_extract):.4f}")
    print(f"  Max: {np.max(feature_extract):.4f}")
    
    if not embeddings_only:
        if os.path.exists(vectorstore_path) and not force_new_embeddings:
            print(f"Loading existing vector store for {dataset_name} with model {model}")
            vectorstore = FAISS.load_local(vectorstore_path, embeddings)
            print(f"Loaded vectorstore with {vectorstore.index.ntotal} vectors")
        else:
            print(f"Creating new vector store for {dataset_name} with model {model}")
            texts = df.loc[selected_indices, content_column].tolist()
            vectorstore = FAISS.from_texts(texts, embeddings, metadatas=[{"index": str(idx)} for idx in selected_indices])
            print(f"Created vectorstore with {vectorstore.index.ntotal} vectors")
        
        # Test vectorstore
        print("Testing vectorstore with a sample query...")
        sample_query = df.loc[selected_indices[0], content_column]
        results = vectorstore.similarity_search_with_score(sample_query, k=5)
        print(f"Sample query: {sample_query[:50]}...")
        for doc, score in results:
            print(f"  Result: {doc.page_content[:50]}..., Score: {score}")
        
        vectorstore.save_local(vectorstore_path)
        print(f"Saved vectorstore to {vectorstore_path}")
    else:
        vectorstore = None
    
    return feature_extract, vectorstore