import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from community import community_louvain
from collections import Counter
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge, Circle

def visualize_multi_category_graph(G, df, id_column, output_file, max_categories=10, min_edge_weight=0.5):
    # Perform community detection
    partition = community_louvain.best_partition(G)
    
    # Set node colors based on community
    unique_communities = sorted(set(partition.values()))
    community_color_map = plt.cm.tab20
    community_colors = {comm: community_color_map(i/len(unique_communities)) for i, comm in enumerate(unique_communities)}
    node_colors = [community_colors[partition[node]] for node in G.nodes()]
    
    # Create a spring layout with more spread
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Create figure and axes
    fig, (ax, cax) = plt.subplots(1, 2, figsize=(24, 20), gridspec_kw={'width_ratios': [20, 1]})
    
    # Draw edges (only those above the minimum weight)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges() if G[u][v]['weight'] >= min_edge_weight]
    if edge_weights:
        max_weight = max(edge_weights)
        min_weight = min(edge_weights)
        if max_weight > min_weight:
            scaled_weights = [(w - min_weight) / (max_weight - min_weight) for w in edge_weights]
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=scaled_weights,
                                   edgelist=[e for e in G.edges() if G[e[0]][e[1]]['weight'] >= min_edge_weight])
        else:
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5,
                                   edgelist=[e for e in G.edges() if G[e[0]][e[1]]['weight'] >= min_edge_weight])
    else:
        print("Warning: No edges meet the minimum weight criteria. The graph will only show disconnected nodes.")
    
    # Define a custom color palette for categories
    custom_colors = [
        '#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4', 
        '#f032e6', '#bfef45', '#fabed4', '#469990', '#dcbeff', '#9A6324', '#fffac8', 
        '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9'
    ]

    # Prepare legend for categories
    all_categories = set()
    for node, data in G.nodes(data=True):
        if 'categories' in data:
            all_categories.update(data['categories'][:max_categories])
    category_counts = Counter(cat for node, data in G.nodes(data=True) 
                              for cat in data.get('categories', [])[:max_categories])
    top_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Assign colors to categories
    category_colors = {cat: custom_colors[i % len(custom_colors)] for i, cat in enumerate(top_categories.keys())}

    # Draw nodes
    for node, (x, y) in pos.items():
        # Draw main node circle (community color)
        circle = Circle((x, y), 0.02, facecolor=node_colors[node], edgecolor='white')
        ax.add_patch(circle)
        
        # Draw category pie chart (smaller, inside the main circle)
        if 'categories' in G.nodes[node]:
            categories = G.nodes[node]['categories'][:max_categories]
            category_counts = Counter(categories)
            total = sum(category_counts.values())
            
            start_angle = 0
            for category, count in category_counts.items():
                angle = 360 * count / total
                wedge = Wedge((x, y), 0.012, start_angle, start_angle+angle, 
                              facecolor=category_colors.get(category, '#808080'), edgecolor='none')
                ax.add_patch(wedge)
                start_angle += angle
    
    # Add labels for the most connected nodes
    top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:10]
    labels = {node: G.nodes[node]['id'] for node, degree in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight="bold")
    
    # Add a color bar to show community assignments
    sm = plt.cm.ScalarMappable(cmap=community_color_map, norm=plt.Normalize(vmin=0, vmax=len(unique_communities)-1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Community', rotation=270, labelpad=15)
    
    # Create legend for top categories
    legend_elements = []
    for category, count in top_categories.items():
        patch = mpatches.Patch(facecolor=category_colors[category], edgecolor='black', label=f"{category} ({count})")
        legend_elements.append(patch)
    
    # Add the category legend to the plot
    ax.legend(handles=legend_elements, title="Top Categories", loc="upper left", bbox_to_anchor=(1, 1))

    # Add explanation for node colors
    ax.text(0.95, 0.05, "Node colors represent different communities\nInner pie charts show category distribution", 
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10, 
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.8))
    
    ax.set_title("Multi-Category Graph Visualization", fontsize=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Graph visualization saved as {output_file}")