import networkx as nx
from sklearn.neighbors import NearestNeighbors
import numpy as np
from community import community_louvain

def classify_with_networkx(feature_extract, df, id_column, category_column, vectorstore=None, n_neighbors=5, initial_threshold=0.7):
    G = nx.Graph()
    
    # Add nodes with IDs and categories as attributes
    for i, (id_value, categories) in enumerate(zip(df[id_column], df[category_column])):
        G.add_node(i, id=id_value, categories=categories.split(',') if isinstance(categories, str) else categories)
    
    all_similarities = []
    
    if vectorstore:
        print("Using vectorstore for similarity search")
        for i, content in enumerate(df[id_column]):
            print(f"Searching for similar items to: {content[:50]}...")
            results = vectorstore.similarity_search_with_score(content, k=n_neighbors+1)
            print(f"Number of results: {len(results)}")
            for doc, score in results[1:]:  # Skip the first result (self)
                matching_indices = df.index[df[id_column] == doc.page_content].tolist()
                if matching_indices:
                    j = matching_indices[0]
                    if i != j:  # Avoid self-loops
                        similarity = 1 / (1 + score)  # Convert distance to similarity
                        all_similarities.append(similarity)
                        G.add_edge(i, j, weight=similarity)
            if i == 0:  # Only print detailed debug for the first item
                print(f"Detailed results for first item:")
                for doc, score in results:
                    similarity = 1 / (1 + score)
                    print(f"  Content: {doc.page_content[:50]}..., Score: {score}, Similarity: {similarity:.4f}")
            if i >= 5:  # Only process the first 5 items for brevity in debugging
                break
    else:
        print("Using sklearn NearestNeighbors for similarity search")
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, metric='cosine').fit(feature_extract)
        distances, indices = nbrs.kneighbors(feature_extract)
        
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            for j, d in zip(idx[1:], dist[1:]):  # Skip the first one (self)
                similarity = 1 - d
                all_similarities.append(similarity)
                G.add_edge(i, j, weight=similarity)
    
    if all_similarities:
        print(f"Similarity score statistics:")
        print(f"  Min: {min(all_similarities):.4f}")
        print(f"  Max: {max(all_similarities):.4f}")
        print(f"  Mean: {np.mean(all_similarities):.4f}")
        print(f"  Median: {np.median(all_similarities):.4f}")
    else:
        print("No similarity scores calculated.")
    
    # Adaptive thresholding
    threshold = initial_threshold
    while G.number_of_edges() == 0 and threshold > 0:
        threshold -= 0.01
        G.remove_edges_from(list(G.edges()))
        G.add_edges_from(((u, v, d) for u, v, d in G.edges(data=True) if d['weight'] > threshold))
    
    if G.number_of_edges() == 0:
        print(f"Warning: No edges in the graph even with minimum threshold. Check your similarity calculations.")
    else:
        print(f"Final edge threshold: {threshold}")
        print(f"Number of edges in the graph: {G.number_of_edges()}")
    
    # Perform community detection
    partition = community_louvain.best_partition(G)
    
    # Add the classification results to the dataframe
    df['networkx_class'] = df.index.map(partition)
    
    return df, G