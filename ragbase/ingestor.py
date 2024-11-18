from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.embeddings import HuggingFaceEmbeddings  # Changed from FastEmbedEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_qdrant import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragbase.config import Config

class Ingestor:
    def __init__(self):
        # Initialize embeddings with error handling
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Initialize text splitters
            self.semantic_splitter = SemanticChunker(
                embeddings=self.embeddings,
                breakpoint_threshold_type="interquartile"
            )
            
            self.recursive_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=128,
                add_start_index=True,
            )
            
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            raise

    def ingest(self, doc_paths: List[Path]) -> VectorStore:
        try:
            documents = []
            for doc_path in doc_paths:
                # Load PDF documents
                loaded_documents = PyPDFium2Loader(str(doc_path)).load()
                
                # Combine document texts
                document_text = "\n".join(
                    [doc.page_content for doc in loaded_documents]
                )
                
                # Split documents using both splitters
                semantic_chunks = self.semantic_splitter.create_documents([document_text])
                final_chunks = self.recursive_splitter.split_documents(semantic_chunks)
                documents.extend(final_chunks)
            
            # Create and return Qdrant vector store
            return Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                path=Config.Path.DATABASE_DIR,
                collection_name=Config.Database.DOCUMENTS_COLLECTION,
            )
            
        except Exception as e:
            print(f"Error in document ingestion: {str(e)}")
            raise