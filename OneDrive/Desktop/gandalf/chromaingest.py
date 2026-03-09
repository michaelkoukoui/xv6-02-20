import hashlib, re, logging
import pandas as pd
import chromadb
import tiktoken
from chromadb.utils import embedding_functions
from typing import List, Dict
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CSV_PATH        = "Advanced_RAG_Dataset_2000.csv"
COLLECTION_NAME = "quantum_docs"
PERSIST_DIR     = "./chroma_db"
EMBED_MODEL     = "all-MiniLM-L6-v2"
MAX_TOKENS      = 256
BATCH_SIZE      = 500

SECTION_ROLES = {
    "change description":        "intent",
    "modifications to baseline": "diff",
    "reasoning":                 "rationale",
    "expert context":            "rationale",
    "legacy compatibility":      "constraint",
    "functional requirements":   "requirement",
    "operational parameters":    "parameter",
    "interface definitions":     "interface",
}

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)
enc = tiktoken.get_encoding("cl100k_base")

# ── Chroma ────────────────────────────────────────────────────────────────────
collection = chromadb.PersistentClient(path=PERSIST_DIR).get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL),
)

# ── Helpers ───────────────────────────────────────────────────────────────────
def uid(text):
    return hashlib.sha256(text.encode()).hexdigest()

def safe(val, default=""):
    return default if pd.isna(val) else str(val).strip()

def classify_role(section_path):
    p = section_path.lower()
    return next((role for kw, role in SECTION_ROLES.items() if kw in p), "content")

def token_split(text):
    """Chunk by token count with 20-token overlap."""
    tokens = enc.encode(text)
    step = MAX_TOKENS - 20
    return [enc.decode(tokens[i:i+MAX_TOKENS]).strip() for i in range(0, len(tokens), step) if tokens[i:i+MAX_TOKENS]]

def parse_sections(text):
    """Split markdown on headings, return list of {section_path, content}."""
    sections, hierarchy, buf, path = [], {1:"",2:"",3:"",4:""}, [], ""
    def flush():
        content = "\n".join(buf).strip()
        if path and content: sections.append({"section_path": path, "content": content})
    for line in text.splitlines():
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush(); buf = []
            lvl, title = len(m.group(1)), m.group(2).strip()
            hierarchy[lvl] = title
            for d in range(lvl+1, 5): hierarchy[d] = ""
            path = " > ".join(hierarchy[i] for i in range(1,5) if hierarchy[i])
        else: buf.append(line)
    flush()
    return sections

def extract_diff_meta(chunk):
    """Pull target_section + change_type out of Modifications bullets.
    FIX: re.search instead of re.match — finds match anywhere in string."""
    m = re.search(r"Section\s+([\d\.]+)\s+\((\w+)\)", chunk)
    return {"target_section": m.group(1), "change_type": m.group(2)} if m else {}

def normalize_version(v):
    """FIX: pandas reads '1' as float 1.0, so '1.0' != '1' breaks supersedes lookup."""
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else str(f)
    except (ValueError, TypeError):
        return str(v).strip()

def resolve_supersedes(doc_id, raw, df):
    if not raw: return ""
    row = df[df["Document ID"] == doc_id]
    if row.empty: return raw
    system = row.iloc[0]["System"]
    match = df[
        (df["System"] == system) &
        (df["Document ID"] != doc_id) &
        (df["Version"].apply(normalize_version) == normalize_version(raw))
    ]
    if not match.empty: return match.iloc[0]["Document ID"]
    log.warning("Could not resolve supersedes='%s' for '%s'", raw, doc_id)
    return raw

def parse_diff_bullets(section_content, base_meta):
    """Each bullet in Modifications section = its own chunk with full metadata."""
    chunks = []
    for line in section_content.splitlines():
        m = re.search(r"Section\s+([\d\.]+)\s+\((\w+)\):\s*(.*)", line)
        if not m:
            continue
        target_sec  = m.group(1)   # "2.2", "3.3", "4.5"
        change_type = m.group(2)   # "Update", "Correction", "New"
        description = m.group(3).strip()

        enriched = (
            f"System: {base_meta['system']} | Doc: {base_meta['doc_id']} | "
            f"Version: {base_meta['version']} | Type: {base_meta['doc_type']} | "
            f"Change to Section {target_sec} ({change_type})\n\n{description}"
        )
        meta = {
            **base_meta,
            "section_role":   "diff",
            "target_section": target_sec,
            "change_type":    change_type,
        }
        chunks.append((enriched, meta))
    return chunks

# ── Ingest ────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
log.info("Loaded %d rows", len(df))

if collection.count() > 0:
    log.info("Collection already has %d chunks, skipping ingest.", collection.count())
else:
    docs, metas, ids = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing"):
        doc_id   = safe(row.get("Document ID"))
        system   = safe(row.get("System"))
        version  = safe(row.get("Version"))
        doc_type = safe(row.get("Document_Type"))
        text     = safe(row.get("Text"))

        if not text:
            log.warning("Skipping '%s' — empty text", doc_id); continue

        supersedes = resolve_supersedes(doc_id, safe(row.get("Supersedes")), df)
        is_delta   = doc_type.upper().startswith("FFD")

        for section in parse_sections(text):
            path = section["section_path"]
            role = classify_role(path)
            
            if role == "diff":
                base_meta = {
                    "system": system, "doc_id": doc_id, "version": version,
                    "supersedes": supersedes, "doc_type": doc_type,
                    "section": path, "is_delta": is_delta,
                }
                for enriched, meta in parse_diff_bullets(section["content"], base_meta):
                    docs.append(enriched)
                    metas.append(meta)
                    ids.append(uid(f"{doc_id}_{version}_{meta['target_section']}_{meta['change_type']}"))
                continue

            for chunk in token_split(section["content"]):
                enriched = (
                    f"System: {system} | Doc: {doc_id} | Version: {version} | "
                    f"Type: {doc_type} | Section: {path} | Role: {role}\n\n{chunk}"
                )
                meta = {
                    "system": system, "doc_id": doc_id, "version": version,
                    "supersedes": supersedes, "doc_type": doc_type,
                    "section": path, "section_role": role, "is_delta": is_delta,
                    **extract_diff_meta(chunk),
                }
                docs.append(enriched)
                metas.append(meta)
                ids.append(uid(f"{doc_id}_{version}_{path}_{chunk}"))

    log.info("Upserting %d chunks", len(docs))
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Upserting"):
        collection.upsert(documents=docs[i:i+BATCH_SIZE], metadatas=metas[i:i+BATCH_SIZE], ids=ids[i:i+BATCH_SIZE])

    log.info("Done. %d chunks indexed.", len(docs))