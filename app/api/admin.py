"""
Admin API endpoints for database initialization and document indexing.
Requires API key authentication via x-api-key header.
"""

import logging
import csv
import io
import threading
import uuid
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from botocore.exceptions import ClientError

from app.api.auth import verify_api_key
from app.core.config import get_settings
from app.core.rag import get_rag_pipeline
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter()




def _get_job_from_db(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status from database."""
    try:
        conn = psycopg2.connect(settings.state_db_url)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT job_id, status, created_at, started_at, completed_at, result, error
            FROM indexing_jobs
            WHERE job_id = %s
        """, (job_id,))
        
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not row:
            return None
        
        def format_timestamp(dt):
            """Format timestamp without timezone and microseconds."""
            if dt is None:
                return None
            # Remove timezone info and microseconds, then format
            return dt.replace(tzinfo=None).replace(microsecond=0).isoformat()
        
        return {
            "job_id": row[0],
            "status": row[1],
            "created_at": format_timestamp(row[2]),
            "started_at": format_timestamp(row[3]),
            "completed_at": format_timestamp(row[4]),
            "result": row[5] if row[5] else None,
            "error": row[6]
        }
    except Exception as e:
        logger.error(f"Failed to get job from database: {e}")
        return None


def _update_job_in_db(job_id: str, status: str, **kwargs):
    """Update job status in database."""
    try:
        conn = psycopg2.connect(settings.state_db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        updates = ["status = %s"]
        values = [status]
        
        if "started_at" in kwargs:
            updates.append("started_at = %s")
            values.append(kwargs["started_at"])
        
        if "completed_at" in kwargs:
            updates.append("completed_at = %s")
            values.append(kwargs["completed_at"])
        
        if "result" in kwargs:
            updates.append("result = %s::jsonb")
            values.append(json.dumps(kwargs["result"]))
        
        if "error" in kwargs:
            updates.append("error = %s")
            values.append(kwargs["error"])
        
        values.append(job_id)
        
        cursor.execute(f"""
            UPDATE indexing_jobs
            SET {', '.join(updates)}
            WHERE job_id = %s
        """, values)
        
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update job in database: {e}")


def _create_job_in_db(job_id: str):
    """Create a new job record in database."""
    try:
        conn = psycopg2.connect(settings.state_db_url)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO indexing_jobs (job_id, status, created_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (job_id) DO NOTHING
        """, (job_id, "queued"))
        
        cursor.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to create job in database: {e}")


def _run_indexing_background(job_id: str):
    """Run indexing in background thread."""
    try:
        _update_job_in_db(job_id, "running", started_at=datetime.utcnow().replace(microsecond=0))
        
        indexer = DocumentIndexer()
        result = indexer.index_documents()
        
        status = "completed" if result.get("success") else "failed"
        _update_job_in_db(
            job_id, 
            status,
            completed_at=datetime.utcnow().replace(microsecond=0),
            result=result
        )
            
    except Exception as e:
        logger.error(f"Background indexing job {job_id} failed: {e}", exc_info=True)
        _update_job_in_db(
            job_id,
            "failed",
            completed_at=datetime.utcnow().replace(microsecond=0),
            error=str(e)
        )


class DatabaseInitializer:
    """Handles database initialization for both vector and state databases."""
    
    def __init__(self):
        """Initialize the database initializer."""
        self.settings = get_settings()
    
    def initialize_vector_database(self) -> bool:
        """Initialize the vector database by creating the vector extension."""
        try:
            logger.info("Initializing vector database...")
            
            conn = psycopg2.connect(self.settings.vector_db_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            logger.info("Creating vector extension...")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            cursor.close()
            conn.close()
            
            logger.info("Vector database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            return False
    
    def initialize_state_database(self) -> bool:
        """Initialize the state database for LangGraph checkpointing."""
        try:
            logger.info("Initializing state database...")
            
            conn = psycopg2.connect(self.settings.state_db_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            logger.info("Creating checkpoints table...")
            cursor.execute("DROP TABLE IF EXISTS checkpoints CASCADE;")
            
            cursor.execute("""
                CREATE TABLE checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    checkpoint JSONB NOT NULL,
                    metadata JSONB,
                    parent_checkpoint_id TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );
            """)
            
            logger.info("Creating state database indexes...")
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_id 
                ON checkpoints (thread_id);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_parent 
                ON checkpoints (parent_checkpoint_id);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_created_at 
                ON checkpoints (created_at);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_checkpoints_thread_ns 
                ON checkpoints (thread_id, checkpoint_ns);
            """)
            
            cursor.execute("""
                CREATE OR REPLACE FUNCTION cleanup_old_checkpoints(days_to_keep INTEGER DEFAULT 30)
                RETURNS INTEGER AS $$
                DECLARE
                    deleted_count INTEGER;
                BEGIN
                    DELETE FROM checkpoints 
                    WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;
                    
                    GET DIAGNOSTICS deleted_count = ROW_COUNT;
                    RETURN deleted_count;
                END;
                $$ LANGUAGE plpgsql;
            """)
            
            logger.info("Creating indexing_jobs table...")
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indexing_jobs (
                    job_id VARCHAR(255) PRIMARY KEY,
                    status VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    started_at TIMESTAMP WITH TIME ZONE,
                    completed_at TIMESTAMP WITH TIME ZONE,
                    result JSONB,
                    error TEXT
                );
            """)
            
            cursor.close()
            conn.close()
            
            logger.info("State database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize state database: {e}")
            return False
    
    def test_connections(self) -> Dict[str, bool]:
        """Test connections to both databases."""
        results = {
            "vector_db": False,
            "state_db": False
        }
        
        try:
            conn = psycopg2.connect(self.settings.vector_db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.close()
            conn.close()
            results["vector_db"] = True
            logger.info("Vector database connection: OK")
        except Exception as e:
            logger.error(f"Vector database connection failed: {e}")
        
        try:
            conn = psycopg2.connect(self.settings.state_db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            cursor.close()
            conn.close()
            results["state_db"] = True
            logger.info("State database connection: OK")
        except Exception as e:
            logger.error(f"State database connection failed: {e}")
        
        return results
    
    def run_initialization(self) -> bool:
        """Run the database initialization process for both vector and state databases."""
        logger.info("Starting database initialization...")
        
        # Test connections
        connections = self.test_connections()
        if not connections["vector_db"]:
            logger.error("Vector database connection failed. Please check your configuration.")
            return False
        if not connections["state_db"]:
            logger.error("State database connection failed. Please check your configuration.")
            return False
        
        # Initialize vector database (create extension)
        if not self.initialize_vector_database():
            logger.error("Vector database initialization failed")
            return False
        
        # Initialize state database (create tables, indexes, functions)
        if not self.initialize_state_database():
            logger.error("State database initialization failed")
            return False
        
        logger.info("Database initialization completed successfully!")
        return True


class DocumentIndexer:
    """Handles indexing of documents from S3-compatible Object Storage."""
    
    def __init__(self):
        """Initialize the document indexer."""
        self.settings = get_settings()
        
        s3_config = Config(
            signature_version='s3v4',
            s3={'addressing_style': 'path'}
        )
        self.s3_client = boto3.client(
            's3',
            endpoint_url=self.settings.linode_object_storage_endpoint,
            aws_access_key_id=self.settings.linode_object_storage_access_key,
            aws_secret_access_key=self.settings.linode_object_storage_secret_key,
            config=s3_config
        )
        
        self.rag_pipeline = get_rag_pipeline()
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all root-level documents in the Object Storage bucket (no subfolders)."""
        try:
            logger.info("Listing root-level documents in Object Storage...")
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.settings.linode_object_storage_bucket,
                Delimiter='/'
            )
            
            documents = []
            # Get root-level objects (not in subfolders)
            # When using Delimiter='/', root-level objects are in 'Contents'
            # and subfolders are in 'CommonPrefixes'
            for obj in response.get('Contents', []):
                key = obj['Key']
                # Only include files at root level (no '/' in the key)
                if '/' not in key:
                    documents.append({
                        "object_key": key,
                        "filename": key,
                        "size": obj['Size'],
                        "last_modified": obj['LastModified'].isoformat() if hasattr(obj['LastModified'], 'isoformat') else str(obj['LastModified']),
                        "etag": obj['ETag']
                    })
            
            logger.info(f"Found {len(documents)} root-level documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def load_metadata_csv(self) -> Dict[str, Dict[str, str]]:
        """
        Load metadata.csv from S3 bucket if it exists.
        
        Returns:
            Dictionary mapping filename to metadata (reference_url, document_date, description, airline)
        """
        metadata = {}
        try:
            
            # Try to get metadata.csv
            try:
                response = self.s3_client.get_object(
                    Bucket=self.settings.linode_object_storage_bucket,
                    Key='metadata.csv'
                )
                
                # Read CSV content
                csv_content = response['Body'].read().decode('utf-8')
                csv_reader = csv.DictReader(io.StringIO(csv_content))
                
                # Map CSV header variations to expected keys
                # Handle both "File Name" and "file_name" formats
                def get_value(row, possible_keys, default=''):
                    for key in possible_keys:
                        value = row.get(key, '').strip()
                        if value:
                            return value
                    return default
                
                for row in csv_reader:
                    filename = get_value(row, ['File Name', 'file_name', 'filename', 'Filename'])
                    if filename:
                        metadata[filename] = {
                            'reference_url': get_value(row, ['Reference URL', 'reference_url', 'ReferenceURL', 'reference-url']),
                            'document_date': get_value(row, ['Document Date', 'document_date', 'DocumentDate', 'document-date']),
                            'description': get_value(row, ['Description', 'description']),
                            'airline': get_value(row, ['Airline', 'airline'])
                        }
                
                logger.info(f"Loaded metadata for {len(metadata)} files from metadata.csv")
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                if error_code == 'NoSuchKey':
                    logger.info("metadata.csv not found in bucket, proceeding without metadata")
                else:
                    logger.warning(f"Failed to load metadata.csv: {e}, proceeding without metadata")
            except Exception as e:
                logger.warning(f"Failed to load metadata.csv: {e}, proceeding without metadata")
                
        except Exception as e:
            logger.warning(f"Error loading metadata.csv: {e}, proceeding without metadata")
        
        return metadata
    
    def clear_all_embeddings(self) -> bool:
        """Clear all existing vector embeddings from the database."""
        try:
            logger.info("Clearing all existing vector embeddings...")
            
            conn = psycopg2.connect(self.settings.vector_db_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if table exists and fix schema if needed
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'langchain_pg_embedding'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                # Check if id column has autoincrement
                cursor.execute("""
                    SELECT column_default 
                    FROM information_schema.columns 
                    WHERE table_name = 'langchain_pg_embedding' 
                    AND column_name = 'id';
                """)
                id_default = cursor.fetchone()
                
                # If id doesn't have a default (autoincrement), fix the schema
                if not id_default or not id_default[0]:
                    logger.info("Fixing langchain_pg_embedding table schema (id column missing autoincrement)...")
                    # Try to alter the column to add autoincrement
                    try:
                        # First, check what type the id column is
                        cursor.execute("""
                            SELECT data_type 
                            FROM information_schema.columns 
                            WHERE table_name = 'langchain_pg_embedding' 
                            AND column_name = 'id';
                        """)
                        id_type = cursor.fetchone()[0]
                        
                        if id_type == 'integer' or id_type == 'bigint':
                            # Drop the table and let PGVector recreate it with correct schema
                            cursor.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
                            logger.info("Dropped langchain_pg_embedding table - PGVector will recreate it with correct schema")
                        else:
                            # If it's not an integer type, drop and recreate
                            cursor.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
                            logger.info("Dropped langchain_pg_embedding table - PGVector will recreate it with correct schema")
                    except Exception as e:
                        logger.warning(f"Could not fix schema automatically: {e}, dropping table")
                        cursor.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;")
                        logger.info("Dropped langchain_pg_embedding table - PGVector will recreate it with correct schema")
                else:
                    # Just delete the data
                    cursor.execute("DELETE FROM langchain_pg_embedding;")
                    deleted_count = cursor.rowcount
                    logger.info(f"Cleared {deleted_count} existing embeddings")
            else:
                logger.info("langchain_pg_embedding table does not exist yet - will be created by PGVector")
            
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear embeddings: {e}")
            return False
    
    def index_documents(self) -> Dict[str, Any]:
        """
        Index all root-level documents from Object Storage into the vector database.
        Clears existing embeddings and replaces them completely.
        Uses metadata.csv if available for additional document metadata.
        """
        try:
            # Clear all existing embeddings first
            if not self.clear_all_embeddings():
                return {
                    "success": False,
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "message": "Failed to clear existing embeddings"
                }
            
            # Load metadata from CSV if available
            metadata_dict = self.load_metadata_csv()
            
            # Get all root-level documents (excluding metadata.csv)
            all_documents = self.list_documents()
            # Filter out metadata.csv from the list of documents to index
            documents_to_index = [doc for doc in all_documents if doc["filename"] != "metadata.csv"]
            
            if not documents_to_index:
                return {
                    "success": True,
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "message": "No documents to index (excluding metadata.csv)"
                }
            
            object_keys = [doc["object_key"] for doc in documents_to_index]
            logger.info(f"Indexing {len(object_keys)} documents (metadata available for {len(metadata_dict)} files)...")
            
            # Index documents with metadata
            result = self.rag_pipeline.index_documents_from_s3(object_keys, metadata_dict=metadata_dict)
            
            logger.info(f"Indexing completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            return {
                "success": False,
                "documents_processed": 0,
                "chunks_created": 0,
                "message": f"Failed to index documents: {str(e)}"
            }
    
    def test_object_storage_connection(self) -> bool:
        """Test connection to Linode Object Storage."""
        try:
            logger.info("Testing Object Storage connection...")
            self.s3_client.head_bucket(Bucket=self.settings.linode_object_storage_bucket)
            logger.info("Object Storage connection: OK")
            return True
        except Exception as e:
            logger.error(f"Object Storage connection failed: {e}")
            return False


# Admin endpoints

@router.post("/admin/initialize_databases")
async def initialize_databases(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Initialize both vector and state databases.
    - Vector database: Creates the vector extension
    - State database: Creates tables, indexes, and functions for LangGraph checkpointing
    Requires API key authentication.
    """
    try:
        logger.info("Database initialization requested via admin API")
        
        initializer = DatabaseInitializer()
        success = initializer.run_initialization()
        
        if success:
            return {
                "success": True,
                "message": "Database initialization completed successfully",
                "vector_db": "initialized",
                "state_db": "initialized"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Database initialization failed. Check logs for details."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database initialization error: {str(e)}"
        )


@router.get("/admin/list_documents")
async def list_documents(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    List all documents in Object Storage.
    Requires API key authentication.
    """
    try:
        logger.info("Document listing requested via admin API")
        
        indexer = DocumentIndexer()
        documents = indexer.list_documents()
        
        return {
            "success": True,
            "count": len(documents),
            "documents": documents
        }
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.post("/admin/index_documents")
async def index_documents(
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Trigger background indexing of all documents.
    Returns immediately with a job ID. Use GET /admin/index_documents/{job_id} to check status.
    Requires API key authentication.
    """
    try:
        logger.info("Document indexing requested via admin API")
        
        # Check for running jobs in database
        conn = psycopg2.connect(settings.state_db_url)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT job_id FROM indexing_jobs
            WHERE status IN ('queued', 'running')
            ORDER BY created_at DESC
            LIMIT 1
        """)
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row:
            return {
                "success": False,
                "message": f"Indexing already in progress. Job ID: {row[0]}",
                "existing_job_id": row[0]
            }
        
        # Create new job
        job_id = str(uuid.uuid4())
        _create_job_in_db(job_id)
        
        # Start background thread
        thread = threading.Thread(target=_run_indexing_background, args=(job_id,))
        thread.daemon = True
        thread.start()
        
        logger.info(f"Started background indexing job: {job_id}")
        
        return {
            "success": True,
            "job_id": job_id,
            "status": "queued",
            "message": "Indexing started in background. Use GET /admin/index_documents/{job_id} to check status."
        }
        
    except Exception as e:
        logger.error(f"Failed to start indexing job: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start indexing job: {str(e)}"
        )


@router.get("/admin/index_documents/{job_id}")
async def get_indexing_status(
    job_id: str,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, Any]:
    """
    Get the status of an indexing job.
    Requires API key authentication.
    """
    job = _get_job_from_db(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job
