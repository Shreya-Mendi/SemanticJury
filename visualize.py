"""
Visualization utilities for semantic space exploration.
Creates interactive plots showing the semantic relationships between legal passages.
"""

import numpy as np
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json

def load_embeddings_and_metadata():
    """Load all embeddings and metadata from ChromaDB."""
    persist_directory = "./chromadb"
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        persist_directory=persist_directory,
        is_persistent=True
    ))
    collection = client.get_collection("legal_cases")

    # Get all documents
    results = collection.get(include=['embeddings', 'metadatas', 'documents'])

    embeddings = np.array(results['embeddings'])
    metadatas = results['metadatas']
    documents = results['documents']

    return embeddings, metadatas, documents

def create_2d_projection(embeddings, method='tsne', perplexity=5):
    """
    Project high-dimensional embeddings to 2D space.

    Args:
        embeddings: Array of embeddings
        method: 'tsne' or 'pca'
        perplexity: t-SNE perplexity parameter (use lower for small datasets)

    Returns:
        2D coordinates
    """
    if method == 'tsne':
        # Adjust perplexity based on dataset size
        n_samples = len(embeddings)
        perplexity = min(perplexity, (n_samples - 1) // 3)

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords_2d = tsne.fit_transform(embeddings)
    else:  # pca
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(embeddings)

    return coords_2d

def create_semantic_space_plot(method='tsne'):
    """
    Create an interactive plot of the semantic space.

    Returns:
        Plotly figure object
    """
    print(f"Loading embeddings...")
    embeddings, metadatas, documents = load_embeddings_and_metadata()

    print(f"Projecting to 2D using {method.upper()}...")
    coords_2d = create_2d_projection(embeddings, method=method)

    # Extract case names and create colors
    case_names = [meta['case_name'] for meta in metadatas]
    unique_cases = list(set(case_names))
    color_map = {case: i for i, case in enumerate(unique_cases)}
    colors = [color_map[name] for name in case_names]

    # Create hover text
    hover_texts = []
    for i, (meta, doc) in enumerate(zip(metadatas, documents)):
        text = f"<b>{meta['case_name']}</b><br>"
        text += f"Position: {meta['position_pct']:.1f}% through opinion<br>"
        text += f"<br>Passage preview:<br>{doc[:200]}..."
        hover_texts.append(text)

    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Case",
                tickvals=list(range(len(unique_cases))),
                ticktext=unique_cases,
                len=0.7
            ),
            line=dict(width=0.5, color='white')
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Legal Passages'
    ))

    fig.update_layout(
        title=f'Semantic Space of Legal Cases ({method.upper()} Projection)',
        xaxis_title=f'{method.upper()} Dimension 1',
        yaxis_title=f'{method.upper()} Dimension 2',
        hovermode='closest',
        width=900,
        height=700,
        plot_bgcolor='rgba(240, 240, 240, 0.9)',
        showlegend=False
    )

    return fig

def create_citation_network_plot():
    """
    Create an interactive network plot showing citation relationships.

    Returns:
        Plotly figure object
    """
    print("Loading citation graph...")
    with open('citation_graph.json', 'r') as f:
        citation_graph = json.load(f)

    # Create nodes
    case_ids = list(citation_graph.keys())
    n_cases = len(case_ids)

    # Simple circular layout
    angles = np.linspace(0, 2*np.pi, n_cases, endpoint=False)
    node_x = np.cos(angles)
    node_y = np.sin(angles)

    # Create edges for citations
    edge_x = []
    edge_y = []

    for i, case_id in enumerate(case_ids):
        cited_cases = citation_graph[case_id].get('cites', [])
        for cited in cited_cases:
            cited_id = cited['case_id']
            if cited_id in case_ids:
                j = case_ids.index(cited_id)
                # Add edge
                edge_x.extend([node_x[i], node_x[j], None])
                edge_y.extend([node_y[i], node_y[j], None])

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='rgba(125, 125, 125, 0.3)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )

    # Create node trace
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=20,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        ),
        text=[cid.replace('_', ' ').title() for cid in case_ids],
        textposition="top center",
        showlegend=False
    )

    # Add hover text with citation info
    hover_texts = []
    for case_id in case_ids:
        cites = citation_graph[case_id].get('cites', [])
        cited_by = citation_graph[case_id].get('cited_by', [])

        text = f"<b>{case_id.replace('_', ' ').title()}</b><br>"
        text += f"Cites: {len(cites)} cases<br>"
        text += f"Cited by: {len(cited_by)} cases"
        hover_texts.append(text)

    node_trace.hovertext = hover_texts

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title='Citation Network of Legal Cases',
        showlegend=False,
        hovermode='closest',
        width=900,
        height=700,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white'
    )

    return fig

if __name__ == "__main__":
    # Test the visualization
    fig = create_semantic_space_plot(method='tsne')
    fig.show()

    fig2 = create_citation_network_plot()
    fig2.show()
