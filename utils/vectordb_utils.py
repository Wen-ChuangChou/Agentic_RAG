import os
import threading
import torch
from concurrent.futures import ThreadPoolExecutor
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from transformers import AutoTokenizer
from typing import List


def sanitize_filename(dataset_name: str) -> str:
    """Convert dataset name to a valid filename by replacing '/' with '_'"""
    return dataset_name.replace("/", "_")


class DocumentProcessor:
    """Thread-safe document processor to avoid pickling issues"""

    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        self.text_splitter = None
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._lock = threading.Lock()

    def _get_text_splitter(self):
        """Initialize text splitter in thread-local manner"""
        if self.text_splitter is None:
            with self._lock:
                if self.text_splitter is None:
                    self.text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                        AutoTokenizer.from_pretrained("thenlper/gte-small"),
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        add_start_index=True,
                        strip_whitespace=True,
                        separators=["\n\n", "\n", ".", " ", ""],
                    )
        return self.text_splitter

    def split_documents_chunk(self,
                              docs_chunk: List[Document]) -> List[Document]:
        """Split a chunk of documents using the text splitter"""
        text_splitter = self._get_text_splitter()
        processed_docs = []
        for doc in docs_chunk:
            new_docs = text_splitter.split_documents([doc])
            processed_docs.extend(new_docs)
        return processed_docs


def parallel_document_splitting(
        source_docs: List[Document],
        max_workers: int = None,
        chunk_size: int = 100,
        text_chunk_size: int = 200,
        text_chunk_overlap: int = 20) -> List[Document]:
    """Split documents in parallel using ThreadPoolExecutor"""
    if max_workers is None:
        max_workers = min(8, len(source_docs) // chunk_size + 1)

    print(f"Splitting documents in parallel using {max_workers} threads...")

    # Split source_docs into chunks for parallel processing
    doc_chunks = [
        source_docs[i:i + chunk_size]
        for i in range(0, len(source_docs), chunk_size)
    ]

    # Create a shared processor instance with configurable chunk size
    processor = DocumentProcessor(text_chunk_size, text_chunk_overlap)

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process chunks in parallel with progress bar
        results = list(
            tqdm(executor.map(processor.split_documents_chunk, doc_chunks),
                 total=len(doc_chunks),
                 desc="Processing document chunks"))

    # Flatten results
    all_processed_docs = []
    for chunk_result in results:
        all_processed_docs.extend(chunk_result)

    return all_processed_docs


def remove_duplicates(docs: List[Document]) -> List[Document]:
    """Remove duplicate documents based on page_content"""
    print("Removing duplicate documents...")
    unique_texts = set()
    unique_docs = []

    for doc in tqdm(docs, desc="Checking for duplicates"):
        if doc.page_content not in unique_texts:
            unique_texts.add(doc.page_content)
            unique_docs.append(doc)

    print(f"Removed {len(docs) - len(unique_docs)} duplicate documents")
    return unique_docs


def batch_embed_documents(docs: List[Document],
                          embedding_model,
                          batch_size: int = 100) -> FAISS:
    """Create FAISS vectorstore with batch processing for embeddings"""
    print(f"Embedding documents in batches of {batch_size}...")

    if not docs:
        raise ValueError("No documents to embed")

    # Create initial vectorstore with first batch
    first_batch = docs[:batch_size]
    print(f"Creating initial vectorstore with {len(first_batch)} documents...")

    vectordb = FAISS.from_documents(
        documents=first_batch,
        embedding=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Process remaining documents in batches
    remaining_docs = docs[batch_size:]
    if remaining_docs:
        total_batches = (len(remaining_docs) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(remaining_docs), batch_size),
                      desc="Adding document batches",
                      total=total_batches):
            batch = remaining_docs[i:i + batch_size]
            vectordb.add_documents(batch)

    return vectordb


def sequential_document_splitting(
        source_docs: List[Document],
        text_chunk_size: int = 200,  # Add this parameter
        text_chunk_overlap: int = 20) -> List[Document]:
    """Fallback sequential document splitting if parallel processing fails"""
    print("Using sequential document splitting...")

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained("thenlper/gte-small"),
        chunk_size=text_chunk_size,
        chunk_overlap=text_chunk_overlap,
        add_start_index=True,
        strip_whitespace=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_processed = []
    for doc in tqdm(source_docs, desc="Splitting documents"):
        new_docs = text_splitter.split_documents([doc])
        docs_processed.extend(new_docs)

    return docs_processed


def load_or_create_vectordb(dataset_name: str,
                            batch_size: int = 100,
                            max_workers: int = None,
                            doc_chunk_size: int = 100,
                            text_chunk_size: int = 200,
                            text_chunk_overlap: int = 20,
                            force_rebuild: bool = False,
                            use_parallel: bool = True) -> FAISS:
    """Load existing vectordb or create new one with optimized processing"""

    # Create sanitized filename
    safe_filename = sanitize_filename(dataset_name)
    vectordb_path = os.path.join(
        "vectordb",
        f"{safe_filename}_{text_chunk_size}")  # Include chunk size in filename

    # Check if vectordb already exists and not forcing rebuild
    if os.path.exists(vectordb_path) and not force_rebuild:
        print(f"Found existing vector database at {vectordb_path}")
        print("Loading existing vector database...")

        try:
            # Load the embedding model (needed for loading)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            embedding_model = HuggingFaceEmbeddings(
                model_name="thenlper/gte-small",
                model_kwargs={"device": device})

            # Load the existing vectordb
            vectordb = FAISS.load_local(vectordb_path,
                                        embedding_model,
                                        allow_dangerous_deserialization=True)
            print("Vector database loaded successfully!")
            return vectordb

        except Exception as e:
            print(f"Error loading existing vectordb: {e}")
            print("Creating new vector database...")

    else:
        if force_rebuild:
            print("Force rebuild requested. Creating new vector database...")
        else:
            print(f"No existing vector database found at {vectordb_path}")
            print("Creating new vector database...")

    # Load and process the dataset
    print("Loading dataset...")
    knowledge_base = datasets.load_dataset(dataset_name, split="train")
    source_docs = [
        Document(page_content=doc["text"],
                 metadata={"source": doc["source"].split("/")[1]})
        for doc in tqdm(knowledge_base, desc="Processing dataset")
    ]

    print(f"Loaded {len(source_docs)} documents from dataset")

    # Split documents (with fallback to sequential processing)
    try:
        if use_parallel and len(source_docs) > 50:
            docs_processed = parallel_document_splitting(
                source_docs, max_workers, doc_chunk_size, text_chunk_size,
                text_chunk_overlap)
        else:
            docs_processed = sequential_document_splitting(
                source_docs, text_chunk_size, text_chunk_overlap)

    except Exception as e:
        print(f"Parallel processing failed: {e}")
        print("Falling back to sequential processing...")
        docs_processed = sequential_document_splitting(
            source_docs, text_chunk_size,
            text_chunk_overlap)  # Pass new parameters

    print(f"Split into {len(docs_processed)} document chunks")

    # Remove duplicates
    docs_processed = remove_duplicates(docs_processed)

    print(f"Final document count after deduplication: {len(docs_processed)}")

    # Create embedding model
    print("Loading embedding model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small",
                                            model_kwargs={"device": device})

    # Create vectordb with batch processing
    vectordb = batch_embed_documents(docs_processed, embedding_model,
                                     batch_size)

    # Save the vectordb
    print(f"Saving vector database to {vectordb_path}...")
    try:
        vectordb.save_local(vectordb_path)
        print("Vector database saved successfully!")
    except Exception as e:
        print(f"Error saving vectordb: {e}")
        print("Continuing without saving...")

    return vectordb
