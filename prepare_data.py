"""
Data preparation script for semantic legal search with citation support.
This script loads legal case data, extracts citations, and prepares it for embedding.
"""

import json
import re
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Citation pattern matcher for legal citations
# Matches formats like: "123 U.S. 456", "789 F.2d 012", etc.
CITATION_PATTERN = r'\b(\d+)\s+([A-Z][a-z]*\.?\s?[A-Z]*\.?[0-9]?[a-z]?)\s+(\d+)\b'

def extract_citations(text: str) -> List[str]:
    """Extract legal citations from text."""
    citations = re.findall(CITATION_PATTERN, text)
    return [f"{vol} {reporter} {page}" for vol, reporter, page in citations]

def chunk_legal_document(case_text: str, case_id: str, case_name: str,
                         chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    Chunk a legal document into passages with provenance tracking.

    Args:
        case_text: Full text of the legal case
        case_id: Unique identifier for the case
        case_name: Name of the case (e.g., "Brown v. Board of Education")
        chunk_size: Maximum number of words per chunk
        overlap: Number of words to overlap between chunks

    Returns:
        List of chunks with metadata
    """
    words = case_text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)

        # Extract citations from this chunk
        citations = extract_citations(chunk_text)

        # Calculate approximate position in document (percentage)
        position_pct = (i / len(words)) * 100 if len(words) > 0 else 0

        chunks.append({
            'text': chunk_text,
            'case_id': case_id,
            'case_name': case_name,
            'chunk_index': len(chunks),
            'position_pct': round(position_pct, 2),
            'citations': citations,
            'word_count': len(chunk_words)
        })

    return chunks

def create_sample_legal_dataset() -> List[Dict]:
    """
    Create a sample dataset of legal cases with citations.
    In production, you would load this from a real legal database.
    """
    cases = [
        {
            'case_id': 'brown_v_board_1954',
            'case_name': 'Brown v. Board of Education, 347 U.S. 483 (1954)',
            'citation': '347 U.S. 483',
            'year': 1954,
            'court': 'Supreme Court',
            'text': '''In Brown v. Board of Education, the Supreme Court held that racial segregation in public schools violated the Equal Protection Clause of the Fourteenth Amendment. The Court overruled Plessy v. Ferguson, 163 U.S. 537 (1896), which had established the "separate but equal" doctrine. Chief Justice Warren wrote that separate educational facilities are inherently unequal. This decision cited to the psychological effects of segregation, referencing studies showing harm to African American children. The Court's reasoning built upon earlier cases like McLaurin v. Oklahoma State Regents, 339 U.S. 637 (1950), and Sweatt v. Painter, 339 U.S. 629 (1950), which had begun to chip away at the separate but equal doctrine in graduate education contexts.'''
        },
        {
            'case_id': 'miranda_v_arizona_1966',
            'case_name': 'Miranda v. Arizona, 384 U.S. 436 (1966)',
            'citation': '384 U.S. 436',
            'year': 1966,
            'court': 'Supreme Court',
            'text': '''Miranda v. Arizona established that defendants must be informed of their rights before custodial interrogation. The Court held that the Fifth Amendment privilege against self-incrimination requires law enforcement to advise suspects of their right to remain silent and to have an attorney present. This case cited Escobedo v. Illinois, 378 U.S. 478 (1964), which had previously recognized the importance of counsel during interrogations. The famous "Miranda warnings" must include notice of the right to remain silent, that statements can be used against the suspect, the right to an attorney, and that an attorney will be appointed if the suspect cannot afford one. The decision built upon the principles established in Gideon v. Wainwright, 372 U.S. 335 (1963), regarding the right to counsel.'''
        },
        {
            'case_id': 'roe_v_wade_1973',
            'case_name': 'Roe v. Wade, 410 U.S. 113 (1973)',
            'citation': '410 U.S. 113',
            'year': 1973,
            'court': 'Supreme Court',
            'text': '''Roe v. Wade recognized a constitutional right to abortion under the Due Process Clause of the Fourteenth Amendment. The Court established a trimester framework for evaluating abortion regulations. Justice Blackmun's majority opinion cited Griswold v. Connecticut, 381 U.S. 479 (1965), which had recognized a right to privacy in marital relations. The decision also referenced Eisenstadt v. Baird, 405 U.S. 438 (1972), extending privacy rights to individuals. The Court balanced the woman's right to privacy against the state's interests in protecting prenatal life and maternal health. This framework was later modified in Planned Parenthood v. Casey, 505 U.S. 833 (1992), which replaced the trimester framework with the undue burden standard.'''
        },
        {
            'case_id': 'marbury_v_madison_1803',
            'case_name': 'Marbury v. Madison, 5 U.S. 137 (1803)',
            'citation': '5 U.S. 137',
            'year': 1803,
            'court': 'Supreme Court',
            'text': '''Marbury v. Madison established the principle of judicial review, affirming the Supreme Court's authority to declare laws unconstitutional. Chief Justice Marshall held that it is the duty of the judicial department to say what the law is. The Court found that Section 13 of the Judiciary Act of 1789 was unconstitutional because it attempted to expand the Court's original jurisdiction beyond what Article III of the Constitution permits. This foundational case created the framework for constitutional interpretation that has guided American jurisprudence for over two centuries. The decision reasoned that a written constitution must be paramount law, and that judges take an oath to uphold the Constitution, requiring them to invalidate conflicting statutes.'''
        },
        {
            'case_id': 'gideon_v_wainwright_1963',
            'case_name': 'Gideon v. Wainwright, 372 U.S. 335 (1963)',
            'citation': '372 U.S. 335',
            'year': 1963,
            'court': 'Supreme Court',
            'text': '''Gideon v. Wainwright held that the Sixth Amendment right to counsel applies to state criminal proceedings through the Fourteenth Amendment's Due Process Clause. The Court overruled Betts v. Brady, 316 U.S. 455 (1942), which had held that appointed counsel was only required in special circumstances. Justice Black wrote that lawyers in criminal courts are necessities, not luxuries. The decision cited Powell v. Alabama, 287 U.S. 45 (1932), which had recognized the importance of counsel in capital cases. This ruling meant that states must provide attorneys to defendants in criminal cases who cannot afford their own lawyers, fundamentally changing the criminal justice system.'''
        },
        {
            'case_id': 'plessy_v_ferguson_1896',
            'case_name': 'Plessy v. Ferguson, 163 U.S. 537 (1896)',
            'citation': '163 U.S. 537',
            'year': 1896,
            'court': 'Supreme Court',
            'text': '''Plessy v. Ferguson upheld racial segregation under the "separate but equal" doctrine. The Court held that a Louisiana law requiring separate railway cars for blacks and whites did not violate the Equal Protection Clause of the Fourteenth Amendment. Justice Brown wrote for the majority that the Fourteenth Amendment was not intended to abolish distinctions based on color or enforce social equality. Justice Harlan's famous dissent argued that the Constitution is color-blind and neither knows nor tolerates classes among citizens. This decision was later overruled by Brown v. Board of Education, 347 U.S. 483 (1954), which held that separate educational facilities are inherently unequal.'''
        }
    ]

    return cases

def build_citation_graph(cases: List[Dict]) -> Dict[str, Dict]:
    """
    Build a citation graph showing which cases cite which other cases.

    Returns:
        Dictionary mapping case_id to citation information
    """
    citation_graph = {}

    # Create lookup by citation string
    citation_lookup = {case['citation']: case['case_id'] for case in cases}

    for case in cases:
        case_id = case['case_id']
        citations_in_text = extract_citations(case['text'])

        # Find which cases are cited
        cited_cases = []
        for citation in citations_in_text:
            if citation in citation_lookup:
                cited_case_id = citation_lookup[citation]
                if cited_case_id != case_id:  # Don't self-reference
                    cited_cases.append({
                        'case_id': cited_case_id,
                        'citation': citation
                    })

        citation_graph[case_id] = {
            'cites': cited_cases,
            'cited_by': []  # Will be populated in next pass
        }

    # Second pass: populate cited_by relationships
    for case_id, info in citation_graph.items():
        for cited in info['cites']:
            cited_id = cited['case_id']
            if cited_id in citation_graph:
                citation_graph[cited_id]['cited_by'].append({
                    'case_id': case_id,
                    'citation': cases[[c['case_id'] for c in cases].index(case_id)]['citation']
                })

    return citation_graph

def prepare_and_store_data(persist_directory: str = "./chromadb"):
    """
    Main function to prepare legal data and store in ChromaDB with citation support.
    """
    print("Loading legal cases...")
    cases = create_sample_legal_dataset()

    print("Building citation graph...")
    citation_graph = build_citation_graph(cases)

    # Save citation graph for later use
    with open('citation_graph.json', 'w') as f:
        json.dump(citation_graph, f, indent=2)

    print("Chunking documents...")
    all_chunks = []
    for case in cases:
        chunks = chunk_legal_document(
            case['text'],
            case['case_id'],
            case['case_name']
        )
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from {len(cases)} cases")

    print("Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("Generating embeddings...")
    texts = [chunk['text'] for chunk in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Storing in ChromaDB...")
    client = chromadb.Client(Settings(
        anonymized_telemetry=False,
        persist_directory=persist_directory,
        is_persistent=True
    ))

    # Delete collection if it exists
    try:
        client.delete_collection("legal_cases")
    except:
        pass

    collection = client.create_collection(
        name="legal_cases",
        metadata={"description": "Legal cases with citation support"}
    )

    # Prepare metadata for each chunk
    metadatas = []
    for chunk in all_chunks:
        metadata = {
            'case_id': chunk['case_id'],
            'case_name': chunk['case_name'],
            'chunk_index': chunk['chunk_index'],
            'position_pct': chunk['position_pct'],
            'citations': json.dumps(chunk['citations']),
            'word_count': chunk['word_count']
        }
        metadatas.append(metadata)

    # Add to collection
    collection.add(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
        ids=[f"{chunk['case_id']}_chunk_{chunk['chunk_index']}" for chunk in all_chunks]
    )

    print(f" Successfully stored {len(all_chunks)} passages in ChromaDB")
    print(f"Citation graph saved to citation_graph.json")

    return collection, citation_graph

if __name__ == "__main__":
    prepare_and_store_data()
