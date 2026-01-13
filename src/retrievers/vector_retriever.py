"""
Unstructured Data Retrieval Channel
Uses vector retrieval technology to retrieve unstructured data (such as documents, images, etc.) and return highly relevant results
"""
import os
import time
import numpy as np
import requests
import json
import logging
from typing import List, Dict, Any, Optional, Union
import concurrent.futures
from threading import Lock
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from ..core.models import RetrievalResult, RetrievalMetadata
from config.config import config
from ..utils.output_manager import output_manager

logger = logging.getLogger('triple_rag')


class OllamaEmbedding:
    """Class for generating embeddings using Ollama API"""

    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", batch_size: int = 10, max_workers: int = 4):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/embeddings"
        self.batch_size = batch_size
        self.max_workers = max_workers
        self._lock = Lock()

    def _get_single_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": text
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result["embedding"]
        except Exception as e:
            print(f"Ollama embedding call failed: {e}")
            # Return zero vector as fallback
            return [0.0] * 768  
    
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a batch of texts concurrently"""
        embeddings = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_text = {executor.submit(self._get_single_embedding, text): text for text in texts}
            for future in concurrent.futures.as_completed(future_to_text):
                try:
                    embedding = future.result()
                    embeddings.append(embedding)
                except Exception as e:
                    text = future_to_text[future]
                    print(f"Failed to process text: {text[:50]}... Error: {e}")
                    embeddings.append([0.0] * 768)
        return embeddings

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embedding vectors for text, supports batch processing"""
        if isinstance(texts, str):
            texts = [texts]

        all_embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        print(f"Starting to process {len(texts)} texts, divided into {total_batches} batches, up to {self.batch_size} per batch")

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")
            start_time = time.time()

            batch_embeddings = self._process_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

            elapsed = time.time() - start_time
            print(f"Batch {batch_num} completed, took {elapsed:.2f} seconds")

        return np.array(all_embeddings)


class VectorRetriever:
    """Vector Retriever"""

    def __init__(self, persist_directory: str = None):
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.llm_client = None  # LLM client for query rewriting
        self.custom_persist_directory = persist_directory  # Custom persist directory
        self.initialize()

    def initialize(self):
        """Initialize vector retriever"""
        try:
            # Initialize LLM client for query rewriting
            self.llm_client = OpenAI(
                base_url=config.llm.base_url,
                api_key=config.llm.api_key
            )
            print(f"LLM client initialized successfully (for query rewriting)")

            # Initialize embedding model
            print(f"Loading embedding model: {config.vector_db.embedding_model}")

            # Determine whether to use ollama embedding
            if config.vector_db.embedding_model.startswith('qwen') or 'ollama' in config.vector_db.embedding_model.lower():
                print("Using Ollama embedding model")
                self.embedding_model = OllamaEmbedding(config.vector_db.embedding_model)
            else:
                print("Using SentenceTransformer embedding model")
                self.embedding_model = SentenceTransformer(config.vector_db.embedding_model)

            print("Embedding model loaded successfully")

            # Initialize ChromaDB client
            if self.custom_persist_directory:
                # Use custom path
                persist_directory = self.custom_persist_directory
            else:
                # Use path from config file
                persist_directory = os.path.join(os.getcwd(), config.vector_db.persist_directory)
            os.makedirs(persist_directory, exist_ok=True)

            self.chroma_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=config.vector_db.collection_name
                )
                print(f"Connected to existing collection: {config.vector_db.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=config.vector_db.collection_name,
                    metadata={"description": "Triple RAG document collection"}
                )
                print(f"Created new collection: {config.vector_db.collection_name}")

        except Exception as e:
            print(f"Vector retriever initialization failed: {e}")

    def rewrite_query_for_vector(
        self,
        original_query: str,
        parent_context: str = None
    ) -> str:
        """
        Use LLM to rewrite query to optimize vector retrieval effectiveness

        Args:
            original_query: Original query text
            parent_context: Parent node answer context (optional)

        Returns:
            Rewritten query text
        """
        if not self.llm_client:
            print("LLM client not initialized, skipping query rewriting")
            return original_query

        try:
            # Build rewriting prompt
            prompt = f"""You are a document retrieval query optimization expert. Your task is to rewrite the user's original query into text more suitable for semantic vector retrieval.

Rewriting requirements:
1. Extract core key entities (company names, product names, locations, times, etc.)
2. Remove all template syntax (such as {{{{p1.answer}}}}, {{{{parent.xxx}}}}, etc.)
3. If there is parent node context, extract key information from it to replace template references
4. Expand into natural English search keywords (as the document library is primarily in English)
5. Preserve time qualifiers (such as "recent", "2024", "latest", etc.)
6. Preserve query intent (such as "news", "trading dynamics", "announcements", etc.)
7. Output concise keyword phrases, not complete sentences, no explanations

Output format: Directly output the rewritten query text, without any prefix or explanation.
"""

            if parent_context:
                prompt += f"\nParent node context:\n{parent_context}\n"

            prompt += f"\nOriginal query:\n{original_query}\n\nRewritten query:"

            # Call LLM for rewriting (removed max_tokens limit)
            response = self.llm_client.chat.completions.create(
                model=config.llm.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional query optimization expert skilled at rewriting complex queries into concise and effective retrieval keywords."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3
            )

            rewritten_query = response.choices[0].message.content.strip()

            # Clean possible extra formatting
            rewritten_query = rewritten_query.strip('"').strip("'").strip()

            # Fallback: If rewriting result is empty, use original query
            if not rewritten_query:
                logger.warning(f"[Query Rewriting] Rewriting result is empty, using original query")
                logger.debug(f"[Query Rewriting] Original: {original_query}")
                return original_query

            logger.info(f"[Query Rewriting] Original: {original_query}")
            logger.info(f"[Query Rewriting] Rewritten: {rewritten_query}")

            return rewritten_query

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}, using original query")
            import traceback
            traceback.print_exc()
            return original_query

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]] = None, ids: List[str] = None):
        """Add documents to vector database"""
        if not self.collection or not self.embedding_model:
            print("Vector retriever not properly initialized")
            return

        try:
            # Generate embedding vectors
            embeddings = self.embedding_model.encode(documents).tolist()

            # Generate IDs (if not provided)
            if not ids:
                ids = [f"doc_{i}" for i in range(len(documents))]

            # Generate metadata (if not provided)
            if not metadatas:
                metadatas = [{"source": "unknown"} for _ in documents]

            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            print(f"Successfully added {len(documents)} documents to vector database")

        except Exception as e:
            print(f"Failed to add documents: {e}")

    def check_existing_ids(self, ids: List[str]) -> List[str]:
        """Check which IDs already exist in the database"""
        if not self.collection:
            return []

        try:
            # Get all existing IDs
            existing_data = self.collection.get(include=["documents"])
            existing_ids = set(existing_data.get('ids', []))

            # Return list of existing IDs
            return [id for id in ids if id in existing_ids]
        except Exception as e:
            print(f"Failed to check existing IDs: {e}")
            return []

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection"""
        if not self.collection:
            return 0

        try:
            result = self.collection.count()
            return result
        except Exception as e:
            print(f"Failed to get collection count: {e}")
            return 0
    
    def search(self, query: str, limit: int = None, parent_context: str = None) -> List[RetrievalResult]:
        """
        Vector retrieval search

        Args:
            query: Query text
            limit: Result count limit
            parent_context: Parent node answer context (for query rewriting)

        Returns:
            List of retrieval results
        """
        if not limit:
            limit = config.max_vector_results

        if not self.collection or not self.embedding_model:
            print("Vector retriever not properly initialized")
            return []

        start_time = time.time()
        results = []

        try:
            # Query Rewriting: All queries are rewritten to extract keywords
            optimized_query = self.rewrite_query_for_vector(query, parent_context)

            # Log output: Vector retrieval query parameters
            output_manager.debug(f"[VECTOR Execution] Query: '{optimized_query}', top_k={limit}")

            # Generate query vector
            query_embedding = self.embedding_model.encode([optimized_query]).tolist()

            # Execute vector search
            search_results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )

            execution_time = time.time() - start_time

            # Create retrieval metadata (record rewritten query)
            metadata = RetrievalMetadata(
                query_statement=f"Vector search: {optimized_query}" + (f" (Original: {query})" if optimized_query != query else ""),
                execution_time=execution_time,
                result_count=len(search_results["documents"][0]) if search_results["documents"] else 0,
                source_type="vector"
            )

            # Convert results to RetrievalResult format
            if search_results["documents"]:
                documents = search_results["documents"][0]
                distances = search_results["distances"][0]
                metadatas = search_results["metadatas"][0]

                for i, (doc, distance, doc_metadata) in enumerate(zip(documents, distances, metadatas)):
                    # Convert distance to similarity score (smaller distance = higher similarity)
                    similarity_score = 1.0 / (1.0 + distance)

                    result = RetrievalResult(
                        content=doc,
                        score=similarity_score,
                        metadata=metadata,
                        source_id=f"vector_doc_{i}",
                        source="vector"
                    )
                    results.append(result)

            print(f"Vector search completed, returned {len(results)} results, took {execution_time:.3f} seconds")

        except Exception as e:
            print(f"Vector search failed: {e}")
            execution_time = time.time() - start_time

            # Record metadata even if failed
            metadata = RetrievalMetadata(
                query_statement=f"Vector search: {query}",
                execution_time=execution_time,
                result_count=0,
                source_type="vector"
            )

            # Return error information as result
            error_result = RetrievalResult(
                content=f"Vector search failed: {str(e)}",
                score=0.0,
                metadata=metadata,
                source_id="vector_error",
                source="vector"
            )
            results.append(error_result)

        return results
    
    def rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results"""
        if not self.embedding_model or len(results) <= 1:
            return results

        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])

            # Generate document embeddings
            documents = [result.content for result in results]
            doc_embeddings = self.embedding_model.encode(documents)

            # Calculate similarities
            similarities = np.dot(query_embedding, doc_embeddings.T)[0]

            # Update scores and sort
            for i, result in enumerate(results):
                result.score = float(similarities[i])

            # Sort by score in descending order
            results.sort(key=lambda x: x.score, reverse=True)

            print(f"Reranking completed, total {len(results)} results")

        except Exception as e:
            print(f"Reranking failed: {e}")

        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        if not self.collection:
            return {}

        try:
            count = self.collection.count()
            return {
                "name": config.vector_db.collection_name,
                "document_count": count,
                "embedding_model": config.vector_db.embedding_model
            }
        except Exception as e:
            print(f"Failed to get collection information: {e}")
            return {}

    def clear_collection(self):
        """Clear collection"""
        if self.collection:
            try:
                # Delete existing collection
                self.chroma_client.delete_collection(config.vector_db.collection_name)

                # Recreate collection
                self.collection = self.chroma_client.create_collection(
                    name=config.vector_db.collection_name,
                    metadata={"description": "Triple RAG document collection"}
                )
                print("Collection cleared and recreated")
            except Exception as e:
                print(f"Failed to clear collection: {e}")
    