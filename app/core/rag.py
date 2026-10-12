"""
RAG (Retrieval-Augmented Generation) pipeline implementation using LangChain.
Handles document indexing, vector storage, and query processing.
"""

import logging
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_postgres import PGVector
from app.core.document_loader import DocumentLoader, S3DocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FastEmbeddings(Embeddings):
    """Lightweight local text embeddings using FastEmbed (ONNX runtime)."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        
    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading FastEmbed model: {self.model_name}")
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return [list(map(float, vec)) for vec in self.model.embed(texts)]
        
    def embed_query(self, text: str) -> List[float]:
        return list(map(float, next(self.model.embed([text]))))


class RAGPipeline:
    """RAG pipeline for document retrieval and question answering."""
    
    def __init__(self):
        """Initialize the RAG pipeline components."""
        self.embeddings = FastEmbeddings(
            model_name=settings.embedding_model
        )
        
        # Initialize ChatGroq with configured model (defaults to openai/gpt-oss-120b)
        groq_key = settings.groq_api_key or "gsk_placeholder"
        self.llm = ChatGroq(
            model=settings.llm_model,
            groq_api_key=groq_key,
            temperature=0.1
        )
        
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None
        
        self._initialize_vector_store()
        self._create_rag_chain()
    
    def _initialize_vector_store(self):
        """Initialize the PostgreSQL vector store."""
        try:
            self.vector_store = PGVector(
                connection=settings.vector_db_url,
                embeddings=self.embeddings,
                collection_name="documents"
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _ensure_table_exists(self):
        """Ensure the langchain_pg_embedding table exists and has correct schema."""
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            
            conn = psycopg2.connect(settings.vector_db_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'langchain_pg_embedding'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                # Table doesn't exist - reinitialize vector store so PGVector will create it
                logger.info("langchain_pg_embedding table does not exist - reinitializing vector store")
                cursor.close()
                conn.close()
                self._initialize_vector_store()
                # Reconnect to check/fix schema after PGVector creates it
                conn = psycopg2.connect(settings.vector_db_url)
                conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                cursor = conn.cursor()
            
            # Always check and fix the id column schema (even if table already exists)
            cursor.execute("""
                SELECT column_default, data_type
                FROM information_schema.columns 
                WHERE table_name = 'langchain_pg_embedding' 
                AND column_name = 'id';
            """)
            id_info = cursor.fetchone()
            
            if id_info:
                id_default, id_type = id_info
                
                # If id doesn't have a default, add appropriate default based on type
                if not id_default:
                    if id_type == 'uuid':
                        logger.info("Fixing langchain_pg_embedding.id column - adding UUID default (gen_random_uuid())")
                        # Ensure pgcrypto extension is enabled for gen_random_uuid()
                        cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
                        cursor.execute("""
                            ALTER TABLE langchain_pg_embedding 
                            ALTER COLUMN id SET DEFAULT gen_random_uuid();
                        """)
                        # Verify the fix was applied
                        cursor.execute("""
                            SELECT column_default
                            FROM information_schema.columns 
                            WHERE table_name = 'langchain_pg_embedding' 
                            AND column_name = 'id';
                        """)
                        verify_default = cursor.fetchone()
                        if verify_default and verify_default[0]:
                            logger.info(f"Successfully added UUID default to id column: {verify_default[0]}")
                        else:
                            logger.error("Failed to add UUID default - column_default is still NULL")
                    elif id_type in ('integer', 'bigint'):
                        logger.info(f"Fixing langchain_pg_embedding.id column - adding sequence default for {id_type}")
                        # For integer types, we need to create a sequence
                        cursor.execute("""
                            CREATE SEQUENCE IF NOT EXISTS langchain_pg_embedding_id_seq;
                            ALTER TABLE langchain_pg_embedding 
                            ALTER COLUMN id SET DEFAULT nextval('langchain_pg_embedding_id_seq');
                            ALTER SEQUENCE langchain_pg_embedding_id_seq OWNED BY langchain_pg_embedding.id;
                        """)
                        logger.info(f"Successfully added sequence default to id column ({id_type})")
                    else:
                        logger.warning(f"id column has no default and is type {id_type} - cannot auto-fix")
            else:
                logger.warning("Could not find id column in langchain_pg_embedding table")
            
            cursor.close()
            conn.close()
                
        except Exception as e:
            logger.error(f"Could not verify/fix table schema: {e}", exc_info=True)
            # Don't raise - let PGVector try to handle it
    
    def _create_rag_chain(self):
        """Create the RAG chain for question answering."""
        try:
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": settings.retrieval_k}
            )
            
            # Define the RAG prompt template
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions based on the provided context. 
                
                Instructions:
                - Answer questions using ONLY the information provided in the context documents
                - Always cite your sources when referencing specific information
                - Include the Airline of the document when citing
                - When possible, include the section of the document (not chunk), as these documents are given section numbers and outlined with clear structure.
                - Don't cite the document number (like "Document 1" or "Document 2") as this is not useful information
                - Don't cite the chunk number (like "Chunk 1" or "Chunk 2") as this is not useful information
                - If the context doesn't contain relevant information, say so clearly
                - Be concise but comprehensive in your answers
                - Maintain a helpful and professional tone
                
                When citing sources, use this format: "According to [File Name] ([Source file])..." or "As mentioned in [File Name]..."."""),
                ("human", "Context:\n{context}\n\nQuestion: {question}")
            ])
            
            # Create the RAG chain using LangChain Expression Language (LCEL)
            self.rag_chain = (
                {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
                | self.prompt_template
                | self.llm
                | StrOutputParser()
            )
            
            logger.info("RAG chain created successfully")
        except Exception as e:
            logger.error(f"Failed to create RAG chain: {e}")
            raise
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents for the prompt with source attribution."""
        formatted_docs = []
        
        for i, doc in enumerate(docs, 1):
            # Extract metadata
            file_name = doc.metadata.get("file_name", "Unknown File")
            source = doc.metadata.get("source", "unknown")
            airline = doc.metadata.get("airline", "")
            reference_url = doc.metadata.get("reference_url", "")
            document_date = doc.metadata.get("document_date", "")
            description = doc.metadata.get("description", "")
            chunk_index = doc.metadata.get("chunk_index", 0)
            total_chunks = doc.metadata.get("total_chunks", 0)
            
            # Format with source information
            doc_header = f"--- {file_name} ---"
            doc_source = f"Source: {source} (chunk {chunk_index + 1} of {total_chunks})"
            if airline:
                doc_source += f"\nAirline: {airline}"
            if reference_url:
                doc_source += f"\nReference URL: {reference_url}"
            if document_date:
                doc_source += f"\nDocument Date: {document_date}"
            if description:
                doc_source += f"\nDescription: {description}"
            doc_content = doc.page_content
            
            formatted_doc = f"{doc_header}\n{doc_source}\n\n{doc_content}"
            formatted_docs.append(formatted_doc)
        
        return "\n\n".join(formatted_docs)
    
    def index_documents(self, object_keys: List[str], metadata_dict: Optional[Dict[str, Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Index documents from the local documents directory.
        
        Args:
            object_keys: List of file names to index
            metadata_dict: Optional dictionary mapping filename to metadata
            
        Returns:
            Dictionary with indexing results
        """
        try:
            # Ensure the table exists (PGVector creates it lazily)
            self._ensure_table_exists()
            
            total_chunks = 0
            processed_docs = 0
            failed_docs = []
            
            loader = DocumentLoader()
            for object_key in object_keys:
                try:
                    logger.info(f"Processing document: {object_key}")
                    documents = loader.load(object_key)
                    
                    if not documents:
                        logger.warning(f"No content found in document: {object_key}")
                        failed_docs.append({"file": object_key, "error": "No content found"})
                        continue
                    
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=settings.chunk_size,
                        chunk_overlap=settings.chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    
                    chunks = text_splitter.split_documents(documents)
                    
                    if not chunks:
                        logger.warning(f"No chunks created from document: {object_key}")
                        failed_docs.append({"file": object_key, "error": "No chunks created"})
                        continue
                    
                    # Get metadata from CSV if available
                    filename = object_key.split('/')[-1]
                    document_metadata = {
                        "file_name": filename,
                        "source": object_key
                    }
                    
                    if metadata_dict and filename in metadata_dict:
                        csv_metadata = metadata_dict[filename]
                        document_metadata.update({
                            "reference_url": csv_metadata.get("reference_url", ""),
                            "document_date": csv_metadata.get("document_date", ""),
                            "description": csv_metadata.get("description", ""),
                            "airline": csv_metadata.get("airline", "")
                        })
                    
                    # Add metadata to chunks
                    for i, chunk in enumerate(chunks):
                        chunk.metadata.update({
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            **document_metadata
                        })
                    
                    # Store chunks in vector database
                    self.vector_store.add_documents(chunks)
                    
                    total_chunks += len(chunks)
                    processed_docs += 1
                    
                    logger.info(f"Successfully indexed {len(chunks)} chunks from {object_key}")
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to index document {object_key}: {error_msg}", exc_info=True)
                    failed_docs.append({"file": object_key, "error": error_msg})
                    # Continue processing other documents
                    continue
            
            # Create vector index for better performance after all documents are added
            if total_chunks > 0:
                logger.info("Creating vector indexes for better search performance...")
                self._create_vector_index()
            
            # Build result message
            if failed_docs:
                message = f"Successfully indexed {processed_docs} documents with {total_chunks} chunks. {len(failed_docs)} document(s) failed: {', '.join([d['file'] for d in failed_docs])}"
            else:
                message = f"Successfully indexed {processed_docs} documents with {total_chunks} chunks"
            
            result = {
                "success": True,
                "documents_processed": processed_docs,
                "chunks_created": total_chunks,
                "failed_documents": failed_docs,
                "message": message
            }
            
            logger.info(f"Document indexing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return {
                "success": False,
                "documents_processed": 0,
                "chunks_created": 0,
                "message": f"Failed to index documents: {str(e)}"
            }
    
    # Backwards compatibility alias
    index_documents_from_s3 = index_documents
    

    def _create_vector_index(self):
        """Create HNSW index for faster vector similarity search."""
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            
            # Connect to database and create index
            conn = psycopg2.connect(settings.vector_db_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # First, check if the embedding column exists and has data
            cursor.execute("""
                SELECT COUNT(*) FROM langchain_pg_embedding WHERE embedding IS NOT NULL;
            """)
            embedding_count = cursor.fetchone()[0]
            
            if embedding_count == 0:
                logger.info("No embeddings found, skipping index creation")
                cursor.close()
                conn.close()
                return
            
            # Check if the embedding column is already a vector type
            cursor.execute("""
                SELECT data_type FROM information_schema.columns 
                WHERE table_name = 'langchain_pg_embedding' AND column_name = 'embedding';
            """)
            column_info = cursor.fetchone()
            
            if not column_info:
                logger.warning("Embedding column not found, skipping index creation")
                cursor.close()
                conn.close()
                return
            
            column_type = column_info[0]
            logger.info(f"Embedding column type: {column_type}")
            
            # If the column is not a vector type, we need to convert it
            if 'vector' not in column_type.lower():
                logger.info("Converting embedding column to vector type...")
                
                # Get the actual embedding dimension from the embedding model
                try:
                    # Create a test embedding to get the correct dimension
                    test_embedding = self.embeddings.embed_query("test")
                    correct_dimension = len(test_embedding)
                    logger.info(f"Using embedding dimension from model: {correct_dimension}")
                except Exception as e:
                    logger.warning(f"Could not get dimension from embedding model: {e}")
                    correct_dimension = 1536  # Default for text-embedding-3-small
                
                # Convert the string representation to proper vector type
                logger.info(f"Converting string embeddings to vector({correct_dimension}) type")
                cursor.execute(f"""
                    ALTER TABLE langchain_pg_embedding 
                    ALTER COLUMN embedding TYPE vector({correct_dimension}) 
                    USING CASE 
                        WHEN embedding::text ~ '^\\[.*\\]$' THEN 
                            embedding::text::vector({correct_dimension})
                        ELSE 
                            NULL::vector({correct_dimension})
                    END;
                """)
            
            # Now create the indexes
            logger.info("Creating HNSW index for vector similarity search...")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS langchain_pg_embedding_embedding_hnsw_idx 
                ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
            
            logger.info("Creating GIN index on metadata for filtering...")
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS langchain_pg_embedding_cmetadata_idx 
                ON langchain_pg_embedding USING gin (cmetadata);
            """)
            
            logger.info("Vector indexes created successfully")
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not create vector index: {e}")
            # Don't raise the exception, just log it as a warning

    def retrieve_docs(self, question: str, airline: Optional[str] = None, k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents, optionally filtered by airline.
        
        Args:
            question: Search query
            airline: Optional airline filter (case-sensitive or normalized)
            k: Number of chunks to retrieve
        """
        k = k or settings.retrieval_k
        if airline and airline.strip() and airline.strip().lower() not in ("all", "all airlines", ""):
            target_airline = airline.strip()
            filter_dict = {"airline": target_airline}
            try:
                docs = self.vector_store.similarity_search(question, k=k, filter=filter_dict)
                if docs:
                    return docs
            except Exception as e:
                logger.warning(f"Filtered retrieval for '{target_airline}' failed: {e}")
        
        # Default unfiltered retrieval
        return self.retriever.invoke(question)

    def query(self, question: str, airline: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the RAG system with a question, optionally filtered by airline.
        
        Args:
            question: The user's question
            airline: Optional airline name to filter documents
            
        Returns:
            Dictionary with the answer, sources, and metadata
        """
        try:
            question_lower = question.lower()
            is_collection_stats_question = any(phrase in question_lower for phrase in [
                'how many airlines', 'how many documents', 'how many policies', 
                'what airlines', 'which airlines', 'list all airlines',
                'number of airlines', 'count of airlines', 'total airlines'
            ])
            
            stats = None
            if is_collection_stats_question:
                stats = self.get_vector_store_stats()
                if stats and stats.get("status") == "connected":
                    airline_count = stats.get('airline_count', 0)
                    airlines = sorted(stats.get('unique_airlines', []))
                    file_count = stats.get('file_count', 0)
                    
                    if 'airline' in question_lower and ('how many' in question_lower or 'number' in question_lower or 'count' in question_lower):
                        if 'list' in question_lower or 'what' in question_lower or 'which' in question_lower:
                            answer = f"I have policy documents for {airline_count} airlines: {', '.join(airlines)}."
                        else:
                            answer = f"I have policy documents for {airline_count} airlines."
                        retrieved_docs = self.retrieve_docs(question, airline=airline)
                    else:
                        stats_info = f"\n\nIMPORTANT: I have policy documents for {airline_count} airlines: {', '.join(airlines)}. Total documents: {file_count}. Use these accurate statistics when answering, not information from individual documents."
                        enhanced_question = f"{question}{stats_info}"
                        retrieved_docs = self.retrieve_docs(question, airline=airline)
                        context = self._format_docs(retrieved_docs)
                        messages = self.prompt_template.format_messages(context=context, question=enhanced_question)
                        answer = self.llm.invoke(messages).content
                else:
                    retrieved_docs = self.retrieve_docs(question, airline=airline)
                    context = self._format_docs(retrieved_docs)
                    messages = self.prompt_template.format_messages(context=context, question=question)
                    answer = self.llm.invoke(messages).content
            else:
                retrieved_docs = self.retrieve_docs(question, airline=airline)
                context = self._format_docs(retrieved_docs)
                messages = self.prompt_template.format_messages(context=context, question=question)
                answer = self.llm.invoke(
                    messages,
                    config={
                        "tags": ["policygpt", "chat"],
                        "metadata": {"airline": airline or "all", "retrieved_count": len(retrieved_docs)}
                    }
                ).content
            
            formatted_sources = [
                {
                    "content": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "file_name": doc.metadata.get("file_name", "Unknown File"),
                    "airline": doc.metadata.get("airline", ""),
                    "reference_url": doc.metadata.get("reference_url", ""),
                    "document_date": doc.metadata.get("document_date", ""),
                    "description": doc.metadata.get("description", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "total_chunks": doc.metadata.get("total_chunks", 0)
                }
                for doc in retrieved_docs
            ]
            
            return {
                "answer": answer,
                "sources": formatted_sources,
                "retrieval_count": len(retrieved_docs)
            }
            
        except Exception as e:
            logger.error(f"Failed to process RAG query: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "sources": [],
                "retrieval_count": 0,
                "error": str(e)
            }

    def stream_query(self, question: str, airline: Optional[str] = None):
        """
        Stream response for the question with retrieved sources.
        
        Returns:
            Tuple of (token_stream_generator, formatted_sources_list)
        """
        retrieved_docs = self.retrieve_docs(question, airline=airline)
        context = self._format_docs(retrieved_docs)
        
        formatted_sources = [
            {
                "content": doc.page_content[:250] + "..." if len(doc.page_content) > 250 else doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "file_name": doc.metadata.get("file_name", "Unknown File"),
                "airline": doc.metadata.get("airline", ""),
                "reference_url": doc.metadata.get("reference_url", ""),
                "document_date": doc.metadata.get("document_date", ""),
                "description": doc.metadata.get("description", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "total_chunks": doc.metadata.get("total_chunks", 0)
            }
            for doc in retrieved_docs
        ]
        
        messages = self.prompt_template.format_messages(context=context, question=question)
        token_stream = self.llm.stream(
            messages,
            config={
                "tags": ["policygpt", "stream"],
                "metadata": {"airline": airline or "all", "retrieved_count": len(retrieved_docs)}
            }
        )
        return token_stream, formatted_sources

    def index_single_document(self, file_name: str, airline: Optional[str] = None, description: Optional[str] = None, reference_url: Optional[str] = None) -> Dict[str, Any]:
        """Index a single document file into vector storage."""
        try:
            self._ensure_table_exists()
            loader = DocumentLoader(settings.documents_dir)
            documents = loader.load_document(file_name)
            if not documents:
                return {"success": False, "message": f"Could not load document {file_name}", "chunks": 0}
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_documents(documents)
            if not chunks:
                return {"success": False, "message": f"No chunks created from {file_name}", "chunks": 0}
            
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "file_name": file_name,
                    "source": file_name,
                    "airline": airline or "",
                    "description": description or "",
                    "reference_url": reference_url or "",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            self.vector_store.add_documents(chunks)
            self._create_vector_index()
            return {
                "success": True,
                "message": f"Successfully indexed {len(chunks)} chunks from {file_name}",
                "chunks": len(chunks)
            }
        except Exception as e:
            logger.error(f"Failed to index single document {file_name}: {e}")
            return {"success": False, "message": str(e), "chunks": 0}

    
    def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store including unique airlines and document counts."""
        try:
            import psycopg2
            from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
            
            conn = psycopg2.connect(settings.vector_db_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Get total chunk count
            cursor.execute("SELECT COUNT(*) FROM langchain_pg_embedding;")
            total_chunks = cursor.fetchone()[0]
            
            # Get unique airlines (from metadata)
            cursor.execute("""
                SELECT DISTINCT cmetadata->>'airline' as airline
                FROM langchain_pg_embedding
                WHERE cmetadata->>'airline' IS NOT NULL 
                AND cmetadata->>'airline' != '';
            """)
            airlines = [row[0] for row in cursor.fetchall() if row[0]]
            
            # Get unique file names
            cursor.execute("""
                SELECT DISTINCT cmetadata->>'file_name' as file_name
                FROM langchain_pg_embedding
                WHERE cmetadata->>'file_name' IS NOT NULL 
                AND cmetadata->>'file_name' != '';
            """)
            file_names = [row[0] for row in cursor.fetchall() if row[0]]
            
            # Get unique documents (by source)
            cursor.execute("""
                SELECT DISTINCT cmetadata->>'source' as source
                FROM langchain_pg_embedding
                WHERE cmetadata->>'source' IS NOT NULL 
                AND cmetadata->>'source' != '';
            """)
            sources = [row[0] for row in cursor.fetchall() if row[0]]
            
            cursor.close()
            conn.close()
            
            return {
                "status": "connected",
                "collection": "documents",
                "total_chunks": total_chunks,
                "unique_airlines": airlines,
                "airline_count": len(airlines),
                "unique_files": file_names,
                "file_count": len(file_names),
                "unique_sources": sources,
                "source_count": len(sources)
            }
        except Exception as e:
            logger.error(f"Failed to get vector store stats: {e}")
            return {
                "status": "error",
                "message": str(e)
            }


# Global RAG pipeline instance
rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """Get the global RAG pipeline instance."""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline


def initialize_rag_pipeline() -> RAGPipeline:
    """Initialize the RAG pipeline."""
    return get_rag_pipeline()
