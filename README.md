# ⚖️ SemanticJury: Legal Search Engine with Citation Support

SemanticJury is a semantic search engine designed for the legal domain. It allows users to query legal texts — such as case law, statutes, and court opinions — using **natural language**, and retrieves passages that are **semantically similar** or **citation-linked** to the query.

## ✨ Features

- 🔍 **Semantic Search**: Find relevant legal passages using natural language queries
- 📚 **Citation Tracking**: Automatically extracts and tracks citations between cases
- 📄 **Provenance**: Every result shows exactly where in the opinion the passage appears
- 🔗 **Citation Network**: Explore which cases cite or are cited by other cases
- 📊 **Citable Passages**: Results include full context and location information for proper citation

## 🛠️ Technical Stack

- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: ChromaDB for efficient similarity search
- **Interface**: Gradio for interactive web UI
- **Language**: Python 3.10+

## 📚 Dataset

This demo includes landmark Supreme Court cases:
- **Brown v. Board of Education** (1954) - School desegregation
- **Miranda v. Arizona** (1966) - Right to remain silent
- **Roe v. Wade** (1973) - Abortion rights
- **Marbury v. Madison** (1803) - Judicial review
- **Gideon v. Wainwright** (1963) - Right to counsel
- **Plessy v. Ferguson** (1896) - Separate but equal doctrine

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SemanticJury.git
cd SemanticJury
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Data

Run the data preparation script to create embeddings and build the citation graph:

```bash
python prepare_data.py
```

This will:
- Load the sample legal cases
- Extract citations between cases
- Chunk documents into passages
- Generate embeddings
- Store everything in ChromaDB
- Create a citation graph (citation_graph.json)

### 4. Launch the App

```bash
python app.py
```

The app will be available at `http://localhost:7860`

## 📖 How to Use

### Semantic Search Tab
1. Enter a natural language query (e.g., "right to counsel in criminal proceedings")
2. Select the number of results you want
3. Click "Search"
4. Results show relevant passages with:
   - Case name and citation
   - Location within the document (% through opinion)
   - The relevant passage text
   - Citations mentioned in that passage
   - Cases that cite or are cited by this case

### Citation Explorer Tab
1. Enter a case ID (e.g., "brown_v_board_1954")
2. Click "Analyze Citations"
3. See:
   - Which cases this case cites
   - Which cases cite this case
   - Full citation information

## 🏗️ Architecture

### Data Pipeline
1. **Legal documents** → Chunked into ~500 word passages with 100 word overlap
2. **Citation extraction** → Regex patterns identify legal citations (e.g., "347 U.S. 483")
3. **Embedding generation** → Each chunk converted to dense vector representation
4. **Storage** → ChromaDB stores embeddings with metadata (case name, position, citations)
5. **Citation graph** → Built to track relationships between cases

### Search Pipeline
1. **Query** → User enters natural language search
2. **Embed query** → Convert query to vector using same model
3. **Similarity search** → Find closest passages using cosine similarity
4. **Enrich results** → Add citation context and provenance information
5. **Display** → Show results with full metadata

## 📁 Project Structure

```
SemanticJury/
├── app.py                  # Main Gradio application
├── prepare_data.py         # Data preparation and embedding script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # License file
├── chromadb/              # Vector database (created by prepare_data.py)
│   └── chroma.sqlite3     # SQLite database file
└── citation_graph.json    # Citation relationships (created by prepare_data.py)
```

## 🔧 Customization

### Adding Your Own Legal Data

Edit `prepare_data.py` and modify the `create_sample_legal_dataset()` function:

```python
cases = [
    {
        'case_id': 'unique_id',
        'case_name': 'Case Name v. Other Party',
        'citation': '123 U.S. 456',
        'year': 2024,
        'court': 'Court Name',
        'text': 'Full text of the case...'
    },
    # Add more cases...
]
```

### Adjusting Chunking Strategy

In `prepare_data.py`, modify the `chunk_legal_document()` function parameters:

```python
def chunk_legal_document(case_text, case_id, case_name,
                         chunk_size=500,  # Adjust chunk size
                         overlap=100):     # Adjust overlap
```

### Using a Different Embedding Model

In both `prepare_data.py` and `app.py`, change the model:

```python
model = SentenceTransformer('your-preferred-model')
```

Options include:
- `sentence-transformers/all-MiniLM-L6-v2` (fast, good quality)
- `sentence-transformers/all-mpnet-base-v2` (slower, better quality)
- `BAAI/bge-large-en-v1.5` (legal-friendly)

## 🚀 Deployment to Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Select "Gradio" as the SDK
3. Push your code:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/SemanticJury
git push hf main
```

4. Make sure your Space includes:
   - `app.py`
   - `requirements.txt`
   - `chromadb/` directory
   - `citation_graph.json`

## 📊 Example Queries

- "constitutional right to privacy"
- "equal protection clause and discrimination"
- "right to remain silent during interrogation"
- "judicial review of legislative acts"
- "separate but equal doctrine"
- "right to appointed counsel"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

See [LICENSE](LICENSE) file for details.

## Technical Implementation

This project demonstrates:
- Semantic search implementation
- Vector database usage
- Natural language processing
- Information retrieval
- Citation network analysis
- Legal text processing

## 🙏 Acknowledgments

- Supreme Court opinions are public domain
- Sentence Transformers library by UKPLab
- ChromaDB by Chroma
- Gradio by Hugging Face

---

**Note**: This is an educational project with a limited dataset. For production legal research, use comprehensive legal databases with proper licensing.
