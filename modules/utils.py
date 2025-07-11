from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import try_to_load_from_cache
import fitz
import json
import requests
import os

def normalize_text(text: str) -> str:
    """Normalize text by replacing non-breaking spaces and stripping whitespace."""
    #NOTE: Might get expanded later when new issues arise
    return text.replace('\u00A0', ' ').strip()


def load_pdf_text(file_path: str) -> str:
    """Extract text from all pages of a PDF using PyMuPDF."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    
    return normalize_text(text)


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Splits text into chunks using LangChain's RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". "]
    )
    return splitter.split_text(text)

def save_indexed_docs(doc_paths: list[str], save_path: str):
    """Saves the indexed document paths to a JSON file."""
    indexed_docs = {"documents": doc_paths}
    path = os.path.join(save_path, "indexed_docs.json")
    with open(path, "w") as f:
        json.dump(indexed_docs, f, indent=4)
    print(f"Indexed document paths saved to indexed_docs.json")

def load_indexed_docs(save_path: str) -> list[str]:
    """Loads the indexed document paths from a JSON file."""
    path = os.path.join(save_path, "indexed_docs.json")
    try:
        with open(path, "r") as f:
            indexed_docs = json.load(f)
        return indexed_docs.get("documents", [])
    except FileNotFoundError:
        print("No indexed documents found. Please build the vector store first.")
        return []
    
def get_model_size(repo_id: str, filename: str = None) -> float:
    """
    Returns size in GB of a specific file (if filename given) or total size of all files in the repo.
    Returns -1.0 on error or if file not found.
    """
    url = f"https://huggingface.co/api/models/{repo_id}?blobs=true"
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        siblings = data.get("siblings", [])
        if filename:
            for f in siblings:
                if f.get("rfilename") == filename:
                    size_bytes = f.get("size", 0) or 0
                    return size_bytes / (1024 ** 3)
            return -1.0
        else:
            total_size = sum(f.get("size", 0) or 0 for f in siblings)
            return total_size / (1024 ** 3)
    except Exception as e:
        print(f"Error fetching model size: {e}")
        return -1.0


def check_model_cache(embedding_model_id: str, llm_repo_id: str, llm_filename: str) -> bool:
    """
    Checks if the embedding model and LLM model are cached.
    Optionally prompts the user if any of them are missing.

    Returns True if safe to proceed, False if user declined download.
    """
    embedding_model_missing = False
    llm_missing = False

    # Check embedding model
    embedding_cache_path = try_to_load_from_cache(embedding_model_id, "config.json")
    if embedding_cache_path and os.path.exists(embedding_cache_path):
        print(f"Embedding model '{embedding_model_id}' is already cached.")
    else:
        print(f"Embedding model '{embedding_model_id}' is not cached.")
        embedding_model_missing = True

    # Check LLM model
    llm_cache_path = try_to_load_from_cache(llm_repo_id, llm_filename)
    if llm_cache_path and os.path.exists(llm_cache_path):
        print(f"LLM file '{llm_filename}' is already cached.")
    else:
        print(f"LLM file '{llm_filename}' is not cached.")
        llm_missing = True

    # Prompt the user if needed
    if embedding_model_missing and llm_missing:
        download_size = get_model_size(embedding_model_id) + get_model_size(llm_repo_id, llm_filename)
        proceed = input(
            f"\nBoth the embedding model '{embedding_model_id}' and the LLM model file '{llm_filename}' are not cached.\n"
            f"This will download {download_size:.2f}GB of data.\n"
            f"Proceed? [y/N]: "
        ).strip().lower()
        return proceed == "y"

    elif embedding_model_missing:
        download_size = get_model_size(embedding_model_id)
        proceed = input(
            f"\nThe embedding model '{embedding_model_id}' is not cached.\n"
            f"This will download {download_size:.2f}GB of data.\n"
            f"Proceed? [y/N]: "
        ).strip().lower()
        return proceed == "y"

    elif llm_missing:
        download_size = get_model_size(llm_repo_id, llm_filename)
        proceed = input(
            f"\nThe LLM model file '{llm_filename}' is not cached.\n"
            f"This will download {download_size:.2f}GB of data.\n"
            f"Proceed? [y/N]: "
        ).strip().lower()
        return proceed == "y"

    return True