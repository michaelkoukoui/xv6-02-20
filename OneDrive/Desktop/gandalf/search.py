import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR     = "./chroma_db"
COLLECTION_NAME = "quantum_docs"
EMBED_MODEL     = "all-MiniLM-L6-v2"

collection = chromadb.PersistentClient(path=PERSIST_DIR).get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL),
)

def search(query, n=5, filters=None):
    results = collection.query(
        query_texts=[query],
        n_results=n,
        where=filters,
    )
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n{'='*60}")
        print(f"Doc:     {meta.get('doc_id')} | Version: {meta.get('version')}")
        print(f"Change:  Section {meta.get('target_section', 'N/A')} | {meta.get('change_type', 'N/A')}")
        print(f"Text:    {doc[:400]}")

query = input("Ask something: ")
search(query, filters={"section_role": "diff"})