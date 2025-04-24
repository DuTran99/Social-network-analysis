import ijson
import json
import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from community import community_louvain
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = []
def process_json_to_csv(json):
    for i, element in enumerate(ijson.items(json, "item")):
        paper = {}
        paper['id'] = element.get('id', np.nan)
        paper['title'] = element.get('title', np.nan)
        paper['year'] = element.get('year', np.nan)

        author = element.get('authors', [])
        author_name = [str(a.get('name', np.nan)) for a in author]
        author_org = [str(a.get('org', np.nan)) for a in author]
        author_id = [str(a.get('id', np.nan)) for a in author]
        paper['author_name'] = ';'.join(author_name)
        paper['author_org'] = ';'.join(author_org)
        paper['author_id'] = ';'.join(author_id)
        paper['n_citation'] = element.get('n_citation', np.nan)
        paper['doc_type'] = element.get('doc_type', np.nan)
        references = element.get('references', [])
        paper['reference_count'] = len(references)
        paper['references'] = ';'.join(map(str, references))
        venue = element.get('venue', {})
        paper['venue_id'] = venue.get('id', np.nan)
        paper['venue_name'] = venue.get('raw', np.nan)
        paper['venue_type'] = venue.get('type', np.nan)
        paper['doi'] = f"https://doi.org/{element.get('doi')}" if element.get('doi') else np.nan
        fos = element.get('fos', [])
        paper['keyword'] = ';'.join([str(f.get('name', np.nan)) for f in fos])
        paper['weight'] = ';'.join([str(f.get('w', np.nan)) for f in fos])
        indexed_abstract = element.get('indexed_abstract', {}).get('InvertedIndex', {})
        paper['indexed_keyword'] = ';'.join(indexed_abstract.keys())
        paper['inverted_index'] = ';'.join(map(str, indexed_abstract.values()))
        paper['publisher'] = element.get('publisher', np.nan)
        paper['volume'] = element.get('volume', np.nan)
        paper['issue'] = element.get('issue', np.nan)
    data = pd.concat([data,paper])
def split_large_json(input_file, output_dir, chunk_size=1000):
    os.makedirs(output_dir, exist_ok=True)

    with open(input_file, 'rb') as f:
        parser = ijson.items(f, 'item') 

        chunk = []
        file_index = 1
        for idx, item in enumerate(parser, start=1):
            chunk.append(item)
            if len(chunk) >= chunk_size:
                process_json_to_csv(chunk)
                print(f'Đã ghi xử lý với {len(chunk)} phần tử')
                chunk = []
                file_index += 1
        if chunk:
            process_json_to_csv(chunk)
            print(f'Đã ghi xử lý với {len(chunk)} phần tử cuối cùng')

split_large_json('C:\Code\dblp.v12.json\dblp.v12.json', 'output_chunks', chunk_size=1000)
# Create a directed graph for citations
G = nx.DiGraph()

# Add nodes (articles) with their attributes
for _, row in data.iterrows():
    G.add_node(row['id'], 
               title=row['title'],
               year=row['year'],
               authors=row['author_name'],
               keywords=row['keyword'],
               venue=row['venue_name'],
               citations=row['n_citation'])

# Add edges (citation relationships)
for _, row in data.iterrows():
    if pd.notna(row['references']):
        references = [int(ref) for ref in str(row['references']).split(';')]
        for ref in references:
            if ref in G.nodes():
                G.add_edge(row['id'], ref)

# Calculate centrality measures
print("\nCalculating centrality measures...")
in_degree_centrality = nx.in_degree_centrality(G)
pagerank = nx.pagerank(G)
betweenness_centrality = nx.betweenness_centrality(G)

# Scale centrality measures for visualization
scaler = MinMaxScaler(feature_range=(5, 20))
in_degree_scaled = scaler.fit_transform(np.array(list(in_degree_centrality.values())).reshape(-1, 1)).flatten()
pagerank_scaled = scaler.fit_transform(np.array(list(pagerank.values())).reshape(-1, 1)).flatten()
betweenness_scaled = scaler.fit_transform(np.array(list(betweenness_centrality.values())).reshape(-1, 1)).flatten()

# Add centrality measures to node attributes
for node in G.nodes():
    G.nodes[node]['in_degree'] = in_degree_centrality[node]
    G.nodes[node]['pagerank'] = pagerank[node]
    G.nodes[node]['betweenness'] = betweenness_centrality[node]
    G.nodes[node]['in_degree_size'] = in_degree_scaled[list(G.nodes()).index(node)]
    G.nodes[node]['pagerank_size'] = pagerank_scaled[list(G.nodes()).index(node)]
    G.nodes[node]['betweenness_size'] = betweenness_scaled[list(G.nodes()).index(node)]

# Community detection using Louvain method (convert to undirected graph first)
undirected_G = G.to_undirected()
partition = community_louvain.best_partition(undirected_G)

# Add community information to node attributes
for node in G.nodes():
    G.nodes[node]['community'] = partition[node]

# Print top nodes by different centrality measures
def print_top_nodes(centrality, name):
    print(f"\nTop 10 nodes by {name}:")
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    for node, value in sorted_nodes:
        title = G.nodes[node]['title'][:50] + '...' if len(G.nodes[node]['title']) > 50 else G.nodes[node]['title']
        print(f"ID: {node}, {name}: {value:.4f}, Title: {title}")

print_top_nodes(in_degree_centrality, "In-Degree Centrality")
print_top_nodes(pagerank, "PageRank")
print_top_nodes(betweenness_centrality, "Betweenness Centrality")

# Visualization
plt.figure(figsize=(20, 15))

# Create a color map for communities
num_communities = max(partition.values()) + 1
colors = plt.cm.tab20(np.linspace(0, 1, num_communities))
node_colors = [colors[partition[node]] for node in G.nodes()]

# Node sizes based on PageRank (you can change to other centrality measures)
node_sizes = [G.nodes[node]['pagerank_size'] * 10 for node in G.nodes()]

# Edge transparency
edge_alphas = [0.1 + 0.9 * (G.nodes[u]['pagerank'] + G.nodes[v]['pagerank'])/2 for u, v in G.edges()]

# Get positions using spring layout
pos = nx.spring_layout(G, k=0.15, iterations=50)

# Draw the graph
nx.draw_networkx_edges(G, pos, alpha=edge_alphas, edge_color='gray', width=0.5)
nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)

# Highlight top 5 nodes by PageRank
top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
top_node_ids = [node[0] for node in top_nodes]
nx.draw_networkx_nodes(G, pos, nodelist=top_node_ids, node_size=[s*1.5 for s in node_sizes[:5]], 
                       node_color=[node_colors[list(G.nodes()).index(node)] for node in top_node_ids], 
                       alpha=1, edgecolors='red', linewidths=2)

# Add labels to top nodes
labels = {node: f"{node}\n{G.nodes[node]['title'][:20]}..." for node in top_node_ids}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')

plt.title("Citation Network with Community Detection and Central Nodes Highlighted", fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.savefig('citation_network_communities.png', dpi=300)

# Save centrality measures to a CSV file for further analysis
centrality_df = pd.DataFrame({
    'id': list(G.nodes()),
    'title': [G.nodes[node]['title'] for node in G.nodes()],
    'year': [G.nodes[node]['year'] for node in G.nodes()],
    'in_degree_centrality': [G.nodes[node]['in_degree'] for node in G.nodes()],
    'pagerank': [G.nodes[node]['pagerank'] for node in G.nodes()],
    'betweenness_centrality': [G.nodes[node]['betweenness'] for node in G.nodes()],
    'community': [G.nodes[node]['community'] for node in G.nodes()],
    'citation_count': [G.nodes[node]['citations'] for node in G.nodes()]
})

centrality_df.to_csv('article_centrality_measures.csv', index=False)
print("\nCentrality measures saved to 'article_centrality_measures.csv'")
