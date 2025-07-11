from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import os

class VectorStoreBuilder:
    """Utility class to build and persist a FAISS vector store from documents."""
    
    def __init__(self, embedding_model, save_path: str):
        self.embedding_model = embedding_model
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
    
    def create_documents_from_chunks(self, chunks: list[str], metadata_list: list[dict] = None) -> list[Document]:
        """Creates LangChain Document objects from text chunks and optional metadata."""
        return [
            Document(page_content=chunk, metadata=metadata_list[i] if metadata_list else {})
            for i, chunk in enumerate(chunks)
        ]
    def build_faiss_index(self, documents: list[Document]) -> FAISS:
        """Embeds and stores documents in FAISS index"""
        db = FAISS.from_documents(documents, self.embedding_model)
        
        db.save_local(self.save_path)
        print(f"FAISS index saved to: {self.save_path}")
        return db
