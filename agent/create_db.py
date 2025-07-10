import os
import shutil

# Document Loading and Processing
from langchain_community.document_loaders import BSHTMLLoader # Use BSHTMLLoader for basic HTML
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
# Embedding and Vector Store
from langchain_google_genai import GoogleGenerativeAIEmbeddings # For embeddings
from langchain_community.vectorstores import Chroma
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('SCANB_AI_DB.log'),
        logging.StreamHandler()
    ]
)

DEFAULT_CORPUS_DIR = 'documents/corpus'
DEFAULT_CHROMA_DB_DIR = 'documents/chroma_db'
DEFAULT_EMBEDDINGS_MODEL = 'models/embedding-001'  # Google Generative AI Embeddings
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 300
K_RETRIEVAL = 2  # Number of documents to retrieve in RAG

class HtmlVectorDatabaseManager:
    """
    Manages the loading, splitting, and vector indexing of an HTML corpus.

    Supports creating a new index or loading an existing one.
    """

    def __init__(
        self,
            google_api_key: str,
        corpus_dir: str = DEFAULT_CORPUS_DIR,
        chroma_db_dir: str = DEFAULT_CHROMA_DB_DIR,
        embeddings_model_name: str = DEFAULT_EMBEDDINGS_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        """
        Initializes the manager with configuration for the corpus and database.

        Args:
            corpus_dir: Directory containing the HTML files.
            chroma_db_dir: Directory to store the ChromaDB index.
            embeddings_model_name: The name of the Google Generative AI embeddings model.
            chunk_size: The size of text chunks for splitting.
            chunk_overlap: The overlap between text chunks.
        """
        self.corpus_dir = corpus_dir
        self.chroma_db_dir = chroma_db_dir
        self.embeddings_model_name = embeddings_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embeddings = GoogleGenerativeAIEmbeddings(model=self.embeddings_model_name, google_api_key=google_api_key)
        self._vector_store: Optional[Chroma] = None # To hold the initialized vector store


    def _load_html_documents(self) -> List[Document]:
        """Loads all HTML files from the configured corpus directory."""
        documents = []
        logging.info(f"Loading documents from {self.corpus_dir}...")
        # Ensure corpus directory exists
        if not os.path.exists(self.corpus_dir):
             logging.error(f"Error: Corpus directory not found at {self.corpus_dir}")
             return []

        for filename in os.listdir(self.corpus_dir):
            if filename.endswith(".html"):
                file_path = os.path.join(self.corpus_dir, filename)
                try:
                    # Use BSHTMLLoader which is part of langchain_community
                    loader = BSHTMLLoader(file_path)
                    docs = loader.load()
                    # Optionally add source metadata
                    for doc in docs:
                         doc.metadata['source'] = file_path
                    documents.extend(docs)
                    logging.info(f"Loaded {file_path}")
                except Exception as e:
                    logging.error(f"Error loading {file_path}: {e}")

        logging.info(f"Finished loading. Total documents loaded: {len(documents)}")
        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits documents into smaller chunks based on configured parameters."""
        if not documents:
            logging.warning("No documents to split.")
            return []

        logging.info(f"Splitting {len(documents)} documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")
        return chunks

    def _create_vector_store(self, chunks: List[Document]):
        """Creates a new Chroma vector store from document chunks and persists it."""
        if not chunks:
             raise ValueError("Cannot create vector store from empty chunks list.")

        print(f"Creating new vector store in {self.chroma_db_dir} from {len(chunks)} chunks...")
        # Ensure the directory exists
        os.makedirs(self.chroma_db_dir, exist_ok=True)

        # Create the new store and persist it
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=self.chroma_db_dir
        )
        # vectorstore.persist() # from_documents with persist_directory often persists automatically, but can call explicitly if needed
        logging.info("New vector store created and persisted.")
        return vectorstore

    def _load_vector_store(self) -> Optional[Chroma]:
        """Loads an existing Chroma vector store from the configured directory."""
        logging.info(f"Attempting to load existing vector store from {self.chroma_db_dir}...")
        # Check if the DB already exists and has files
        if not os.path.exists(self.chroma_db_dir) or not os.listdir(self.chroma_db_dir):
             logging.warning("No existing vector store found.")
             return None

        try:
            vectorstore = Chroma(persist_directory=self.chroma_db_dir, embedding_function=self._embeddings)
            # Optional: Add a quick check to see if it loaded correctly (e.g., count items)
            # count = vectorstore._collection.count()
            # logging.info(f"Vector store loaded successfully with {count} items.")
            logging.info("Vector store loaded.")
            return vectorstore
        except Exception as e:
            logging.error(f"Error loading vector store from {self.chroma_db_dir}: {e}")
            return None


    def initialize_vector_store(self, force_reindex: bool = False) -> Optional[Chroma]:
        """
        Initializes the vector store by loading an existing one or creating a new one
        from the corpus if it doesn't exist or force_reindex is True.

        Args:
            force_reindex: If True, forces rebuilding the index even if one exists.

        Returns:
            The initialized Chroma vector store instance, or None if initialization failed.
        """
        if self._vector_store and not force_reindex:
            logging.info("Vector store already initialized.")
            return self._vector_store

        logging.info("\n--- Initializing Vector Store ---")

        if force_reindex:
            logging.info(f"Force reindex requested. Clearing existing Chroma DB at {self.chroma_db_dir}...")
            if os.path.exists(self.chroma_db_dir):
                 shutil.rmtree(self.chroma_db_dir)

        # Attempt to load existing store first unless force_reindex is true
        if not force_reindex:
            loaded_store = self._load_vector_store()
            if loaded_store:
                self._vector_store = loaded_store
                return self._vector_store

        # If load failed or force_reindex is true, create a new one
        logging.info("Existing vector store not found or reindexing forced. Building new one...")
        html_docs = self._load_html_documents()
        if not html_docs:
            logging.warning(f"No documents found in {self.corpus_dir} to build the index.")
            return None # Cannot build if no documents

        html_chunks = self._split_documents(html_docs)
        if not html_chunks:
            logging.warning("No chunks created after splitting. Cannot build the index.")
            return None
        try:
            self._vector_store = self._create_vector_store(html_chunks)
            return self._vector_store
        except Exception as e:
            logging.error(f"Failed to create vector store: {e}")
            self._vector_store = None
            return None

    def get_vector_store(self):
        """
        Returns the initialized vector store.

        Call initialize_vector_store() first.
        """
        if not self._vector_store:
            logging.warning("Vector store not initialized. Call initialize_vector_store() first.")
        return self._vector_store

    def get_retriever(self, k: int = K_RETRIEVAL) -> Optional[VectorStore]:
         """
         Returns a retriever instance from the initialized vector store.

         Args:
             k: The number of documents to retrieve.

         Call initialize_vector_store() first.
         """
         if self._vector_store:
              return self._vector_store.as_retriever(search_kwargs={"k": k})
         else:
              print("Vector store not initialized. Cannot get retriever.")
              return None