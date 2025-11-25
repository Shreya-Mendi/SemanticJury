"""
Semantic Legal Search Engine with Citation Support
Supports finding cases that cite or are cited by other cases,
and surfaces citable passages with full provenance.
"""

import json
import gradio as gr
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Tuple
import os
from visualize import create_semantic_space_plot, create_citation_network_plot

# Initialize the embedding model
print("Loading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load ChromaDB
print("Loading ChromaDB...")
persist_directory = "./chromadb"
if not os.path.exists(persist_directory):
    print("ChromaDB not found. Please run prepare_data.py first!")
else:
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        persist_directory=persist_directory,
        is_persistent=True
    ))
    collection = client.get_collection("legal_cases")
    print(f"Loaded collection with {collection.count()} passages")

# Load citation graph
try:
    with open('citation_graph.json', 'r') as f:
        citation_graph = json.load(f)
    print("✅ Loaded citation graph")
except FileNotFoundError:
    print("Warning: Citation graph not found. Please run prepare_data.py first!")
    citation_graph = {}

def semantic_search(query: str, n_results: int = 5) -> List[Dict]:
    """
    Perform semantic search over legal cases.

    Args:
        query: Search query
        n_results: Number of results to return

    Returns:
        List of search results with metadata
    """
    query_embedding = model.encode([query])
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=n_results
    )

    formatted_results = []
    for i in range(len(results['documents'][0])):
        metadata = results['metadatas'][0][i]
        formatted_results.append({
            'text': results['documents'][0][i],
            'case_name': metadata['case_name'],
            'case_id': metadata['case_id'],
            'chunk_index': metadata['chunk_index'],
            'position_pct': metadata['position_pct'],
            'citations': json.loads(metadata['citations']),
            'distance': results['distances'][0][i] if 'distances' in results else None
        })

    return formatted_results

def find_citing_cases(case_id: str) -> List[Dict]:
    """Find all cases that cite the given case."""
    if case_id not in citation_graph:
        return []

    citing_cases = citation_graph[case_id].get('cited_by', [])
    return citing_cases

def find_cited_cases(case_id: str) -> List[Dict]:
    """Find all cases cited by the given case."""
    if case_id not in citation_graph:
        return []

    cited_cases = citation_graph[case_id].get('cites', [])
    return cited_cases

def format_result_with_provenance(result: Dict, rank: int) -> str:
    """Format a search result with full provenance information."""
    output = f"### Result {rank}\n\n"
    output += f"**Case:** {result['case_name']}\n\n"
    output += f"**Location in Document:** ~{result['position_pct']:.1f}% through the opinion\n\n"
    output += f"**Relevant Passage:**\n\n"
    output += f"> {result['text']}\n\n"

    if result['citations']:
        output += f"**Citations in this passage:** {', '.join(result['citations'])}\n\n"

    # Show citation relationships
    case_id = result['case_id']

    cited_cases = find_cited_cases(case_id)
    if cited_cases:
        output += f"**This case cites:** "
        citations_list = [f"{c['citation']}" for c in cited_cases]
        output += ", ".join(citations_list) + "\n\n"

    citing_cases = find_citing_cases(case_id)
    if citing_cases:
        output += f"**This case is cited by:** "
        citations_list = [f"{c['citation']}" for c in citing_cases]
        output += ", ".join(citations_list) + "\n\n"

    output += "---\n\n"
    return output

def search_interface(query: str, num_results: int = 5) -> str:
    """Main search interface function."""
    if not query.strip():
        return "Warning: Please enter a search query."

    try:
        results = semantic_search(query, n_results=num_results)

        if not results:
            return "No results found."

        output = f"# Search Results for: \"{query}\"\n\n"
        output += f"Found {len(results)} relevant passages:\n\n"
        output += "---\n\n"

        for i, result in enumerate(results, 1):
            output += format_result_with_provenance(result, i)

        return output

    except Exception as e:
        return f"Error: {str(e)}\n\nPlease make sure you've run prepare_data.py first!"

def citation_search_interface(case_name: str) -> str:
    """Search for cases by citation relationships."""
    if not case_name.strip():
        return "Warning: Please enter a case name or ID."

    # Try to find case ID from name
    case_id = None
    for cid in citation_graph.keys():
        if case_name.lower() in cid.lower():
            case_id = cid
            break

    if not case_id:
        return f"Error: Case not found: {case_name}\n\nAvailable cases:\n" + "\n".join(
            [f"- {cid}" for cid in citation_graph.keys()]
        )

    output = f"# Citation Analysis for: {case_id}\n\n"

    # Cases cited by this case
    cited = find_cited_cases(case_id)
    output += f"## This case cites ({len(cited)} cases):\n\n"
    if cited:
        for c in cited:
            output += f"- **{c['citation']}** (ID: {c['case_id']})\n"
    else:
        output += "- None found\n"

    output += "\n"

    # Cases that cite this case
    citing = find_citing_cases(case_id)
    output += f"## This case is cited by ({len(citing)} cases):\n\n"
    if citing:
        for c in citing:
            output += f"- **{c['citation']}** (ID: {c['case_id']})\n"
    else:
        output += "- None found\n"

    return output

def get_case_context(case_id: str, position_pct: float, context_window: int = 3) -> str:
    """Get surrounding passages for context."""
    # Query for passages from the same case near this position
    results = collection.get(
        where={
            "case_id": case_id,
            "position_pct": {"$gte": position_pct - 10, "$lte": position_pct + 10}
        },
        limit=context_window
    )

    if not results['documents']:
        return "No additional context available."

    output = "### Surrounding Context:\n\n"
    for i, doc in enumerate(results['documents']):
        metadata = results['metadatas'][i]
        output += f"**Chunk at {metadata['position_pct']:.1f}%:**\n{doc}\n\n"

    return output

# Create Gradio interface
with gr.Blocks(title="Semantic Legal Search with Citations") as demo:
    gr.Markdown("""
    # Semantic Legal Search Engine
    ## Search Supreme Court Cases with Citation Support

    This tool allows you to:
    - **Semantically search** through legal opinions
    - **Track citations** between cases
    - **View citable passages** with full provenance
    - **Explore citation networks**
    """)

    with gr.Tab("Semantic Search"):
        gr.Markdown("### Search for legal concepts, doctrines, or specific issues")
        with gr.Row():
            with gr.Column():
                search_input = gr.Textbox(
                    label="Search Query",
                    placeholder="e.g., right to counsel, equal protection, privacy rights",
                    lines=2
                )
                num_results = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Number of Results"
                )
                search_btn = gr.Button("Search", variant="primary")

            with gr.Column():
                gr.Markdown("""
                **Example Queries:**
                - "right to counsel in criminal proceedings"
                - "equal protection and education"
                - "privacy rights and due process"
                - "judicial review and constitutional interpretation"
                """)

        search_output = gr.Markdown(label="Search Results")

        search_btn.click(
            fn=search_interface,
            inputs=[search_input, num_results],
            outputs=search_output
        )

        # Example queries
        gr.Examples(
            examples=[
                ["right to counsel in criminal proceedings", 5],
                ["equal protection and racial segregation", 5],
                ["privacy rights and due process", 3],
                ["judicial review constitutional", 5],
            ],
            inputs=[search_input, num_results],
        )

    with gr.Tab("Citation Explorer"):
        gr.Markdown("### Explore which cases cite or are cited by other cases")
        with gr.Row():
            with gr.Column():
                citation_input = gr.Textbox(
                    label="Case Name or ID",
                    placeholder="e.g., brown_v_board_1954",
                    lines=1
                )
                citation_btn = gr.Button("Analyze Citations", variant="primary")

            with gr.Column():
                gr.Markdown("""
                **Available Cases:**
                - brown_v_board_1954
                - miranda_v_arizona_1966
                - roe_v_wade_1973
                - marbury_v_madison_1803
                - gideon_v_wainwright_1963
                - plessy_v_ferguson_1896
                """)

        citation_output = gr.Markdown(label="Citation Analysis")

        citation_btn.click(
            fn=citation_search_interface,
            inputs=citation_input,
            outputs=citation_output
        )

        gr.Examples(
            examples=[
                "brown_v_board_1954",
                "miranda_v_arizona_1966",
                "gideon_v_wainwright_1963"
            ],
            inputs=citation_input,
        )

    with gr.Tab("Semantic Space Visualization"):
        gr.Markdown("### Explore the semantic relationships between legal passages")

        with gr.Row():
            projection_method = gr.Radio(
                choices=["tsne", "pca"],
                value="tsne",
                label="Projection Method",
                info="t-SNE preserves local structure, PCA is faster"
            )
            visualize_btn = gr.Button("Generate Visualization", variant="primary")

        semantic_plot = gr.Plot(label="Semantic Space")
        citation_network_plot = gr.Plot(label="Citation Network")

        def generate_visualizations(method):
            try:
                semantic_fig = create_semantic_space_plot(method=method)
                citation_fig = create_citation_network_plot()
                return semantic_fig, citation_fig
            except Exception as e:
                import plotly.graph_objects as go
                error_fig = go.Figure()
                error_fig.add_annotation(
                    text=f"Error: {str(e)}<br>Make sure you've run prepare_data.py first!",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return error_fig, error_fig

        visualize_btn.click(
            fn=generate_visualizations,
            inputs=projection_method,
            outputs=[semantic_plot, citation_network_plot]
        )

        gr.Markdown("""
        **How to interpret:**
        - **Semantic Space Plot**: Points close together are semantically similar passages
        - **Citation Network**: Lines show which cases cite each other
        - Colors represent different cases
        - Hover over points for details
        """)

if __name__ == "__main__":
    demo.launch(share=False)
