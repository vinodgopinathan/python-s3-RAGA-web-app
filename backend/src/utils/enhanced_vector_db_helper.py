import os
import uuid
import logging
import psycopg2
import psycopg2.extras
import psycopg2.pool
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from datetime import datetime
import json
import threading
import time

logger = logging.getLogger(__name__)

class PooledConnection:
    """Wrapper for pooled database connections to ensure proper return to pool"""
    def __init__(self, conn, pool):
        self.conn = conn
        self.pool = pool
        self._closed = False
    
    def __enter__(self):
        return self.conn
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def close(self):
        if not self._closed and self.pool is not None:
            try:
                self.pool.putconn(self.conn)
                self._closed = True
            except Exception as e:
                logger.warning(f"Error returning connection to pool: {e}")
    
    def __getattr__(self, name):
        return getattr(self.conn, name)

class EnhancedVectorDBHelper:
    def __init__(self):
        """Initialize enhanced vector database helper with connection pooling and better transaction management"""
        # Support both POSTGRES_* and DB_* environment variables for flexibility
        self.host = os.environ.get('DB_HOST') or os.environ.get('POSTGRES_HOST', 'localhost')
        self.port = int(os.environ.get('DB_PORT') or os.environ.get('POSTGRES_PORT', '5432'))
        self.database = os.environ.get('DB_NAME') or os.environ.get('POSTGRES_DB', 'rag_system')
        self.user = os.environ.get('DB_USER') or os.environ.get('POSTGRES_USER', 'postgres')
        self.password = os.environ.get('DB_PASSWORD') or os.environ.get('POSTGRES_PASSWORD', '')
        
        # Initialize connection pool for better concurrent access
        self._connection_pool = None
        self._pool_lock = threading.Lock()
        self._initialize_connection_pool()
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384
        
        # Initialize database schema only if needed and not skipped
        skip_db_init = os.environ.get('SKIP_DB_INIT', 'false').lower() == 'true'
        if not skip_db_init:
            self._ensure_schema_exists()
        else:
            logger.info("Database initialization skipped (SKIP_DB_INIT=true)")
    
    def _initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool for better concurrent access"""
        try:
            # Create connection pool with appropriate settings for AWS RDS
            self._connection_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,    # Minimum connections
                maxconn=20,   # Maximum connections (good for AWS RDS)
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                # Connection pool settings
                application_name='aws-llm-raga-chunking',
                # Timeout settings for better reliability
                connect_timeout=30,
                keepalives_idle=600,
                keepalives_interval=30,
                keepalives_count=3
            )
            logger.info("Database connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            # Fallback to direct connections
            self._connection_pool = None
    
    def _ensure_schema_exists(self):
        """Check if required tables exist and create schema only if needed"""
        try:
            import os
            force_recreate = os.environ.get('FORCE_DB_RECREATE', 'false').lower() == 'true'
            
            if force_recreate:
                logger.info("FORCE_DB_RECREATE=true. Recreating database schema...")
                self._create_schema()
                return
            
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Check if main tables exist
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name IN ('documents', 'document_chunks')
            """)
            
            table_count = cur.fetchone()[0]
            
            # If tables don't exist, create schema
            if table_count < 2:
                logger.info("Database tables not found. Creating schema...")
                self._create_schema()
            else:
                logger.info("Database schema already exists. Skipping creation.")
                
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not check schema existence: {e}. Will attempt to create schema.")
            self._create_schema()
    
    def _create_schema(self):
        """Create database schema from SQL file"""
        try:
            import os
            
            # Look for schema file in multiple locations
            schema_paths = [
                'backend/database_schema.sql',
                'database_schema.sql',
                '../database_schema.sql',
                os.path.join(os.path.dirname(__file__), '../../database_schema.sql')
            ]
            
            schema_file = None
            for path in schema_paths:
                if os.path.exists(path):
                    schema_file = path
                    break
            
            if not schema_file:
                logger.error("Database schema file not found. Please ensure database_schema.sql exists.")
                return False
            
            conn = self._get_connection()
            cur = conn.cursor()
            
            # First, drop existing tables if they exist (for FORCE_DB_RECREATE)
            force_recreate = os.environ.get('FORCE_DB_RECREATE', 'false').lower() == 'true'
            if force_recreate:
                logger.info("Dropping existing tables for schema recreation...")
                try:
                    cur.execute("DROP TABLE IF EXISTS document_chunks CASCADE;")
                    cur.execute("DROP TABLE IF EXISTS documents CASCADE;")
                    conn.commit()
                    logger.info("Existing tables dropped successfully")
                except Exception as e:
                    logger.warning(f"Could not drop existing tables: {e}")
            
            # Read and execute schema
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema creation
            cur.execute(schema_sql)
            conn.commit()
            
            cur.close()
            conn.close()
            
            logger.info("Database schema created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create database schema: {e}")
            return False
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test database connection"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def _get_connection(self):
        """Get database connection from pool or create direct connection"""
        if self._connection_pool is not None:
            try:
                # Get connection from pool with timeout
                with self._pool_lock:
                    conn = self._connection_pool.getconn()
                    return PooledConnection(conn, self._connection_pool)
            except Exception as e:
                logger.warning(f"Failed to get pooled connection: {e}, falling back to direct connection")
        
        # Fallback to direct connection
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
            application_name='aws-llm-raga-chunking',
            connect_timeout=30
        )
    
    def create_document_record(self, 
                             file_path: str,
                             file_name: str,
                             file_type: str,
                             source_type: str,
                             chunking_method: str,
                             s3_key: Optional[str] = None,
                             file_size: Optional[int] = None,
                             content_type: Optional[str] = None,
                             chunk_size: int = 1000,
                             chunk_overlap: int = 200) -> str:
        """Create a new document record and return document ID"""
        
        document_id = str(uuid.uuid4())
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO documents 
                    (id, file_path, s3_key, file_name, file_type, file_size, 
                     content_type, source_type, chunking_method, chunk_size, 
                     chunk_overlap, processing_status, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    document_id, file_path, s3_key, file_name, file_type,
                    file_size, content_type, source_type, chunking_method,
                    chunk_size, chunk_overlap, 'processing', datetime.utcnow()
                ))
                conn.commit()
        
        logger.info(f"Created document record: {document_id} - {file_name}")
        return document_id
    
    def update_document_status(self, 
                             document_id: str, 
                             status: str, 
                             total_chunks: Optional[int] = None,
                             error_message: Optional[str] = None):
        """Update document processing status"""
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if status == 'completed':
                    cur.execute("""
                        UPDATE documents 
                        SET processing_status = %s, total_chunks = %s, 
                            processed_at = %s, updated_at = %s
                        WHERE id = %s
                    """, (status, total_chunks, datetime.utcnow(), datetime.utcnow(), document_id))
                elif status == 'failed':
                    cur.execute("""
                        UPDATE documents 
                        SET processing_status = %s, error_message = %s, updated_at = %s
                        WHERE id = %s
                    """, (status, error_message, datetime.utcnow(), document_id))
                else:
                    cur.execute("""
                        UPDATE documents 
                        SET processing_status = %s, updated_at = %s
                        WHERE id = %s
                    """, (status, datetime.utcnow(), document_id))
                
                conn.commit()
    
    def store_document_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
        """Store document chunks with embeddings and metadata using batch transactions for better reliability
        
        Returns:
            Tuple of (stored_count, failed_chunks_list)
        """
        
        stored_count = 0
        failed_chunks = []
        
        logger.info(f"CHUNK_STORAGE_START - Document {document_id}: Attempting to store {len(chunks)} chunks")
        
        # Process chunks in batches to avoid connection/transaction issues
        batch_size = 5  # Smaller batches to reduce transaction pressure
        
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]
            
            logger.debug(f"Processing chunk batch {batch_start+1}-{batch_end} for document {document_id}")
            
            # Process entire batch in single transaction
            try:
                with self._get_connection() as conn:
                    conn.autocommit = False
                    batch_success = True
                    batch_failures = []
                    
                    try:
                        with conn.cursor() as cur:
                            for i, chunk in enumerate(batch_chunks):
                                chunk_index = chunk.get('chunk_index', batch_start + i)
                                
                                try:
                                    # Check if chunk already exists (prevent duplicates)
                                    cur.execute(
                                        "SELECT id FROM document_chunks WHERE document_id = %s AND chunk_index = %s",
                                        (document_id, chunk_index)
                                    )
                                    existing_chunk = cur.fetchone()
                                    
                                    if existing_chunk:
                                        logger.debug(f"Chunk {chunk_index} already exists for document {document_id}, skipping")
                                        stored_count += 1  # Count as stored since it exists
                                        continue
                            
                                    # Validate chunk content
                                    content = chunk.get('content', '')
                                    if not content or not content.strip():
                                        error_msg = f"Chunk {chunk_index} has empty content"
                                        logger.warning(f"CHUNK_STORAGE_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                                        batch_failures.append({
                                            'chunk_index': chunk_index,
                                            'error_type': 'empty_content',
                                            'error_message': error_msg,
                                            'content_preview': str(content)[:100] if content else 'None'
                                        })
                                        batch_success = False
                                        continue
                                    
                                    # Generate embedding with error handling
                                    try:
                                        embedding = self.embedding_model.encode(content)
                                        embedding_list = embedding.tolist()
                                    except Exception as embedding_error:
                                        error_msg = f"Failed to generate embedding: {str(embedding_error)}"
                                        logger.error(f"CHUNK_EMBEDDING_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                                        batch_failures.append({
                                            'chunk_index': chunk_index,
                                            'error_type': 'embedding_generation',
                                            'error_message': error_msg,
                                            'content_preview': content[:100] if content else 'None',
                                            'content_length': len(content)
                                        })
                                        batch_success = False
                                        continue
                                    
                                    # Extract metadata
                                    metadata = chunk.get('metadata', {})
                                    chunking_method = metadata.get('chunking_method', 'unknown')
                                    
                                    # Prepare chunk data with validation
                                    try:
                                        chunk_data = {
                                            'id': str(uuid.uuid4()),
                                            'document_id': document_id,
                                            'chunk_index': chunk_index,
                                            'content': content,
                                            'content_length': len(content),
                                            'embedding': embedding_list,
                                            'chunking_method': chunking_method,
                                            'separator_used': metadata.get('separator_used'),
                                            'char_count': metadata.get('char_count', len(content)),
                                            'sentence_count': metadata.get('sentence_count'),
                                            'semantic_boundary': metadata.get('semantic_boundary'),
                                            'similarity_score': metadata.get('similarity_score'),
                                            'llm_reasoning': metadata.get('llm_reasoning'),
                                            'llm_confidence': metadata.get('llm_confidence'),
                                            'llm_provider': metadata.get('llm_provider'),
                                            'adaptive_method': metadata.get('adaptive_method'),
                                            'text_analysis': json.dumps(metadata.get('text_analysis', {})),
                                            'chosen_strategy': metadata.get('chosen_strategy'),
                                            'metadata': json.dumps(metadata),
                                            'source_type': metadata.get('source_type', 'unknown'),
                                            's3_key': metadata.get('s3_key'),
                                            'file_path': metadata.get('file_path'),
                                            'content_type': metadata.get('content_type')
                                        }
                                    except Exception as data_prep_error:
                                        error_msg = f"Failed to prepare chunk data: {str(data_prep_error)}"
                                        logger.error(f"CHUNK_DATA_PREP_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                                        batch_failures.append({
                                            'chunk_index': chunk_index,
                                            'error_type': 'data_preparation',
                                            'error_message': error_msg,
                                            'content_preview': content[:100] if content else 'None',
                                            'metadata_keys': list(metadata.keys()) if metadata else []
                                        })
                                        batch_success = False
                                        continue
                                    
                                    # Insert chunk with detailed error handling
                                    try:
                                        cur.execute("""
                                            INSERT INTO document_chunks 
                                            (id, document_id, chunk_index, content, content_length, embedding,
                                             chunking_method, separator_used, char_count, sentence_count,
                                             semantic_boundary, similarity_score, llm_reasoning, llm_confidence,
                                             llm_provider, adaptive_method, text_analysis, chosen_strategy,
                                             metadata, source_type, s3_key, file_path, content_type, created_at)
                                            VALUES (%(id)s, %(document_id)s, %(chunk_index)s, %(content)s, 
                                                   %(content_length)s, %(embedding)s, %(chunking_method)s,
                                                   %(separator_used)s, %(char_count)s, %(sentence_count)s,
                                                   %(semantic_boundary)s, %(similarity_score)s, %(llm_reasoning)s,
                                                   %(llm_confidence)s, %(llm_provider)s, %(adaptive_method)s,
                                                   %(text_analysis)s, %(chosen_strategy)s, %(metadata)s,
                                                   %(source_type)s, %(s3_key)s, %(file_path)s, %(content_type)s,
                                                   %(created_at)s)
                                        """, {**chunk_data, 'created_at': datetime.utcnow()})
                                        
                                        logger.debug(f"Successfully prepared chunk {chunk_index} for batch commit")
                                        
                                    except psycopg2.IntegrityError as integrity_error:
                                        error_msg = f"Database integrity error: {str(integrity_error)}"
                                        logger.error(f"CHUNK_INTEGRITY_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                                        batch_failures.append({
                                            'chunk_index': chunk_index,
                                            'error_type': 'database_integrity',
                                            'error_message': error_msg,
                                            'content_preview': content[:100],
                                            'sql_error_code': getattr(integrity_error, 'pgcode', 'unknown')
                                        })
                                        batch_success = False
                                        continue
                                        
                                    except psycopg2.DataError as data_error:
                                        error_msg = f"Database data error: {str(data_error)}"
                                        logger.error(f"CHUNK_DATA_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                                        batch_failures.append({
                                            'chunk_index': chunk_index,
                                            'error_type': 'database_data',
                                            'error_message': error_msg,
                                            'content_preview': content[:100],
                                            'content_length': len(content),
                                            'embedding_length': len(embedding_list) if 'embedding_list' in locals() else 0
                                        })
                                        batch_success = False
                                        continue
                                        
                                except Exception as chunk_error:
                                    error_msg = f"Chunk processing error: {str(chunk_error)}"
                                    logger.error(f"CHUNK_PROCESSING_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                                    batch_failures.append({
                                        'chunk_index': chunk_index,
                                        'error_type': 'chunk_processing',
                                        'error_message': error_msg,
                                        'content_preview': content[:100] if 'content' in locals() else 'unknown',
                                        'exception_type': type(chunk_error).__name__
                                    })
                                    batch_success = False
                                    continue
                            
                            # Commit entire batch if all chunks processed successfully
                            if batch_success and not batch_failures:
                                conn.commit()
                                batch_stored = len(batch_chunks) - len([c for c in batch_chunks if 'existing' in str(c)])
                                stored_count += batch_stored
                                logger.debug(f"Successfully committed batch {batch_start+1}-{batch_end} ({batch_stored} chunks)")
                            else:
                                conn.rollback()
                                logger.warning(f"Rolling back batch {batch_start+1}-{batch_end} due to {len(batch_failures)} failures")
                                failed_chunks.extend(batch_failures)
                                
                    except Exception as batch_error:
                        conn.rollback()
                        error_msg = f"Batch processing error: {str(batch_error)}"
                        logger.error(f"BATCH_PROCESSING_ERROR - Document {document_id}, Batch {batch_start+1}-{batch_end}: {error_msg}")
                        # Mark all chunks in batch as failed
                        for i, chunk in enumerate(batch_chunks):
                            chunk_index = chunk.get('chunk_index', batch_start + i)
                            failed_chunks.append({
                                'chunk_index': chunk_index,
                                'error_type': 'batch_processing',
                                'error_message': error_msg,
                                'exception_type': type(batch_error).__name__
                            })
                            
            except Exception as connection_error:
                error_msg = f"Database connection error: {str(connection_error)}"
                logger.error(f"BATCH_CONNECTION_ERROR - Document {document_id}, Batch {batch_start+1}-{batch_end}: {error_msg}")
                # Mark all chunks in batch as failed
                for i, chunk in enumerate(batch_chunks):
                    chunk_index = chunk.get('chunk_index', batch_start + i)
                    failed_chunks.append({
                        'chunk_index': chunk_index,
                        'error_type': 'database_connection',
                        'error_message': error_msg,
                        'exception_type': type(connection_error).__name__
                    })
                                    'source_type': metadata.get('source_type', 'unknown'),
                                    's3_key': metadata.get('s3_key'),
                                    'file_path': metadata.get('file_path'),
                                    'content_type': metadata.get('content_type')
                                }
                            except Exception as data_prep_error:
                                error_msg = f"Failed to prepare chunk data: {str(data_prep_error)}"
                                logger.error(f"CHUNK_DATA_PREP_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                                failed_chunks.append({
                                    'chunk_index': chunk_index,
                                    'error_type': 'data_preparation',
                                    'error_message': error_msg,
                                    'content_preview': content[:100] if content else 'None',
                                    'metadata_keys': list(metadata.keys()) if metadata else []
                                })
                                continue
                            
                            # Insert chunk with detailed error handling
                            try:
                                cur.execute("""
                                    INSERT INTO document_chunks 
                                    (id, document_id, chunk_index, content, content_length, embedding,
                                     chunking_method, separator_used, char_count, sentence_count,
                                     semantic_boundary, similarity_score, llm_reasoning, llm_confidence,
                                     llm_provider, adaptive_method, text_analysis, chosen_strategy,
                                     metadata, source_type, s3_key, file_path, content_type, created_at)
                                    VALUES (%(id)s, %(document_id)s, %(chunk_index)s, %(content)s, 
                                           %(content_length)s, %(embedding)s, %(chunking_method)s,
                                           %(separator_used)s, %(char_count)s, %(sentence_count)s,
                                           %(semantic_boundary)s, %(similarity_score)s, %(llm_reasoning)s,
                                           %(llm_confidence)s, %(llm_provider)s, %(adaptive_method)s,
                                           %(text_analysis)s, %(chosen_strategy)s, %(metadata)s,
                                           %(source_type)s, %(s3_key)s, %(file_path)s, %(content_type)s,
                                           %(created_at)s)
                                """, {**chunk_data, 'created_at': datetime.utcnow()})
                                
                                conn.commit()
                                stored_count += 1
                                logger.debug(f"Successfully stored chunk {chunk_index} for document {document_id}")
                                
                            except psycopg2.IntegrityError as integrity_error:
                                conn.rollback()
                                error_msg = f"Database integrity error: {str(integrity_error)}"
                                logger.error(f"CHUNK_INTEGRITY_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                                failed_chunks.append({
                                    'chunk_index': chunk_index,
                                    'error_type': 'database_integrity',
                                    'error_message': error_msg,
                                    'content_preview': content[:100],
                                    'sql_error_code': getattr(integrity_error, 'pgcode', 'unknown')
                                })
                                continue
                                
                            except psycopg2.DataError as data_error:
                                conn.rollback()
                                error_msg = f"Database data error: {str(data_error)}"
                                logger.error(f"CHUNK_DATA_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                                failed_chunks.append({
                                    'chunk_index': chunk_index,
                                    'error_type': 'database_data',
                                    'error_message': error_msg,
                                    'content_preview': content[:100],
                                    'content_length': len(content),
                                    'embedding_length': len(embedding_list) if 'embedding_list' in locals() else 0
                                })
                                continue
                            
                    except Exception as chunk_error:
                        conn.rollback()
                        error_msg = f"Chunk processing error: {str(chunk_error)}"
                        logger.error(f"CHUNK_PROCESSING_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                        failed_chunks.append({
                            'chunk_index': chunk_index,
                            'error_type': 'chunk_processing',
                            'error_message': error_msg,
                            'content_preview': content[:100] if 'content' in locals() else 'unknown',
                            'exception_type': type(chunk_error).__name__
                        })
                        continue
                        
            except Exception as connection_error:
                error_msg = f"Database connection error: {str(connection_error)}"
                logger.error(f"CHUNK_CONNECTION_ERROR - Document {document_id}, Chunk {chunk_index}: {error_msg}")
                failed_chunks.append({
                    'chunk_index': chunk_index,
                    'error_type': 'database_connection',
                    'error_message': error_msg,
                    'exception_type': type(connection_error).__name__
                })
                continue
        
        # Log comprehensive summary
        if failed_chunks:
            logger.warning(f"CHUNK_STORAGE_SUMMARY - Document {document_id}: {stored_count}/{len(chunks)} chunks stored successfully, {len(failed_chunks)} chunks failed")
            
            # Group failures by error type for better analysis
            error_summary = {}
            for failed_chunk in failed_chunks:
                error_type = failed_chunk['error_type']
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
            
            logger.warning(f"CHUNK_FAILURE_BREAKDOWN - Document {document_id}: {error_summary}")
            
            # Store chunk failures in processing_errors table for tracking
            self._log_chunk_failures_to_database(document_id, failed_chunks, stored_count, len(chunks))
        else:
            logger.info(f"CHUNK_STORAGE_SUCCESS - Document {document_id}: All {stored_count}/{len(chunks)} chunks stored successfully")
        
        # CRITICAL: Validate actual stored chunks vs expected chunks
        # This catches silent failures that weren't detected during processing
        actual_stored_chunks = self._validate_chunk_storage(document_id, len(chunks))
        if actual_stored_chunks != len(chunks):
            logger.error(f"CHUNK_STORAGE_VALIDATION_FAILED - Document {document_id}: Expected {len(chunks)} chunks, but only {actual_stored_chunks} found in database")
            
            # Find which chunks are actually missing
            missing_chunks = self._find_missing_chunks(document_id, len(chunks))
            if missing_chunks:
                logger.error(f"CHUNK_STORAGE_MISSING_CHUNKS - Document {document_id}: Missing chunk indices: {missing_chunks}")
                
                # Create failed_chunks entries for missing chunks that weren't detected
                additional_failures = []
                for missing_index in missing_chunks:
                    additional_failures.append({
                        'chunk_index': missing_index,
                        'error_type': 'silent_storage_failure',
                        'error_message': f'Chunk {missing_index} was not found in database after storage attempt',
                        'content_preview': 'Unknown - chunk processing appeared successful but storage failed',
                        'detection_method': 'post_storage_validation'
                    })
                
                # Log these additional failures to the database
                if additional_failures:
                    logger.warning(f"CHUNK_STORAGE_SILENT_FAILURES - Document {document_id}: Detected {len(additional_failures)} silent chunk storage failures")
                    self._log_chunk_failures_to_database(document_id, additional_failures, actual_stored_chunks, len(chunks))
                    
                    # Update failed_chunks list and stored_count to reflect reality
                    failed_chunks.extend(additional_failures)
                    stored_count = actual_stored_chunks
        
        return stored_count, failed_chunks
    
    def _log_chunk_failures_to_database(self, document_id: str, failed_chunks: List[Dict[str, Any]], stored_count: int, total_chunks: int):
        """Log chunk failures to processing_errors table for tracking and analysis"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Get document info for context
                    cur.execute(
                        "SELECT file_name, file_path, s3_key FROM documents WHERE id = %s",
                        (document_id,)
                    )
                    doc_info = cur.fetchone()
                    
                    if not doc_info:
                        logger.warning(f"Could not find document {document_id} for error logging")
                        return
                    
                    file_name, file_path, s3_key = doc_info
                    
                    # Map chunk error types to processing error types
                    error_type_mapping = {
                        'empty_content': 'validation_error',
                        'embedding_generation': 'embedding_error',
                        'data_preparation': 'chunking_error',
                        'database_integrity': 'database_error',
                        'database_data': 'database_error',
                        'chunk_processing': 'chunking_error',
                        'database_connection': 'database_error'
                    }
                    
                    # Determine processing stage
                    processing_stage_mapping = {
                        'empty_content': 'validation',
                        'embedding_generation': 'embedding_generation',
                        'data_preparation': 'text_chunking',
                        'database_integrity': 'database_storage',
                        'database_data': 'database_storage',
                        'chunk_processing': 'text_chunking',
                        'database_connection': 'database_storage'
                    }
                    
                    # Group failed chunks by error type to reduce database entries
                    grouped_failures = {}
                    for failed_chunk in failed_chunks:
                        error_type = failed_chunk['error_type']
                        if error_type not in grouped_failures:
                            grouped_failures[error_type] = {
                                'count': 0,
                                'chunk_indices': [],
                                'sample_error': failed_chunk['error_message'],
                                'details': []
                            }
                        
                        grouped_failures[error_type]['count'] += 1
                        grouped_failures[error_type]['chunk_indices'].append(failed_chunk['chunk_index'])
                        grouped_failures[error_type]['details'].append({
                            'chunk_index': failed_chunk['chunk_index'],
                            'error_message': failed_chunk['error_message'],
                            'content_preview': failed_chunk.get('content_preview', ''),
                            'additional_info': {k: v for k, v in failed_chunk.items() 
                                             if k not in ['chunk_index', 'error_message', 'content_preview', 'error_type']}
                        })
                    
                    # Insert grouped error records
                    for error_type, failure_info in grouped_failures.items():
                        mapped_error_type = error_type_mapping.get(error_type, 'other')
                        mapped_stage = processing_stage_mapping.get(error_type, 'database_storage')
                        
                        error_message = f"Failed to store {failure_info['count']} chunk(s) due to {error_type}: {failure_info['sample_error']}"
                        
                        error_details = {
                            'chunk_error_type': error_type,
                            'failed_chunk_count': failure_info['count'],
                            'failed_chunk_indices': failure_info['chunk_indices'],
                            'chunk_details': failure_info['details'][:10],  # Limit details to avoid huge records
                            'total_chunks_in_document': total_chunks,
                            'successfully_stored_chunks': stored_count
                        }
                        
                        cur.execute("""
                            INSERT INTO processing_errors 
                            (document_id, file_name, file_path, s3_key, failure_scope, error_type, 
                             error_message, error_details, processing_stage, retry_count, 
                             is_recoverable, resolved)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            document_id, file_name, file_path, s3_key, 'chunk', mapped_error_type,
                            error_message, json.dumps(error_details), mapped_stage, 0,
                            error_type not in ['database_integrity', 'database_data'],  # Some errors might be recoverable
                            False  # Not resolved yet
                        ))
                    
                    conn.commit()
                    logger.info(f"Logged {len(grouped_failures)} chunk error types to processing_errors table for document {document_id}")
                    
        except Exception as e:
            logger.error(f"Failed to log chunk failures to database for document {document_id}: {e}")
    
    def _validate_chunk_storage(self, document_id: str, expected_chunks: int) -> int:
        """Validate how many chunks are actually stored in the database for a document"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM document_chunks WHERE document_id = %s",
                        (document_id,)
                    )
                    result = cur.fetchone()
                    return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to validate chunk storage for document {document_id}: {e}")
            return 0
    
    def _find_missing_chunks(self, document_id: str, expected_chunks: int) -> List[int]:
        """Find which chunk indices are missing from the database"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT chunk_index FROM document_chunks WHERE document_id = %s ORDER BY chunk_index",
                        (document_id,)
                    )
                    stored_indices = [row[0] for row in cur.fetchall()]
                    expected_indices = list(range(expected_chunks))
                    missing_indices = [i for i in expected_indices if i not in stored_indices]
                    return missing_indices
        except Exception as e:
            logger.error(f"Failed to find missing chunks for document {document_id}: {e}")
            return []
    
    def get_chunk_failure_analytics(self) -> Dict[str, Any]:
        """Get analytics on chunk-level failures"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Get chunk failure summary (only chunk-level failures)
                    cur.execute("""
                        SELECT 
                            error_type,
                            processing_stage,
                            COUNT(*) as error_count,
                            SUM((error_details->>'failed_chunk_count')::int) as total_failed_chunks,
                            AVG((error_details->>'failed_chunk_count')::int) as avg_failed_chunks_per_error,
                            COUNT(CASE WHEN resolved = true THEN 1 END) as resolved_errors,
                            MIN(created_at) as first_occurrence,
                            MAX(created_at) as last_occurrence
                        FROM processing_errors 
                        WHERE failure_scope = 'chunk'
                        GROUP BY error_type, processing_stage
                        ORDER BY total_failed_chunks DESC
                    """)
                    
                    chunk_failures = [dict(row) for row in cur.fetchall()]
                    
                    # Get document-level failure summary for comparison
                    cur.execute("""
                        SELECT 
                            error_type,
                            processing_stage,
                            COUNT(*) as error_count,
                            COUNT(CASE WHEN resolved = true THEN 1 END) as resolved_errors,
                            MIN(created_at) as first_occurrence,
                            MAX(created_at) as last_occurrence
                        FROM processing_errors 
                        WHERE failure_scope = 'document'
                        GROUP BY error_type, processing_stage
                        ORDER BY error_count DESC
                    """)
                    
                    document_failures = [dict(row) for row in cur.fetchall()]
                    
                    # Get recent chunk failures
                    cur.execute("""
                        SELECT 
                            file_name,
                            error_type,
                            error_message,
                            error_details->>'failed_chunk_count' as failed_chunks,
                            error_details->>'successfully_stored_chunks' as stored_chunks,
                            error_details->>'total_chunks_in_document' as total_chunks,
                            created_at
                        FROM processing_errors 
                        WHERE failure_scope = 'chunk'
                        ORDER BY created_at DESC
                        LIMIT 20
                    """)
                    
                    recent_chunk_failures = [dict(row) for row in cur.fetchall()]
                    
                    # Get recent document failures
                    cur.execute("""
                        SELECT 
                            file_name,
                            error_type,
                            error_message,
                            processing_stage,
                            created_at
                        FROM processing_errors 
                        WHERE failure_scope = 'document'
                        ORDER BY created_at DESC
                        LIMIT 20
                    """)
                    
                    recent_document_failures = [dict(row) for row in cur.fetchall()]
                    
                    return {
                        'chunk_failure_summary': chunk_failures,
                        'document_failure_summary': document_failures,
                        'recent_chunk_failures': recent_chunk_failures,
                        'recent_document_failures': recent_document_failures
                    }
                    
        except Exception as e:
            logger.error(f"Error getting chunk failure analytics: {e}")
            return {
                'chunk_failure_summary': [], 
                'document_failure_summary': [],
                'recent_chunk_failures': [],
                'recent_document_failures': []
            }
    
    def get_failure_breakdown_by_scope(self) -> Dict[str, Any]:
        """Get a breakdown of failures by scope (document vs chunk)"""
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    # Overall breakdown by scope
                    cur.execute("""
                        SELECT 
                            failure_scope,
                            COUNT(*) as total_errors,
                            COUNT(DISTINCT document_id) as affected_documents,
                            COUNT(CASE WHEN resolved = true THEN 1 END) as resolved_errors,
                            COUNT(CASE WHEN resolved = false THEN 1 END) as unresolved_errors,
                            COALESCE(SUM((error_details->>'failed_chunk_count')::int), 0) as total_failed_chunks
                        FROM processing_errors 
                        GROUP BY failure_scope
                        ORDER BY total_errors DESC
                    """)
                    
                    scope_breakdown = [dict(row) for row in cur.fetchall()]
                    
                    # Error type distribution by scope
                    cur.execute("""
                        SELECT 
                            failure_scope,
                            error_type,
                            COUNT(*) as error_count,
                            COUNT(DISTINCT document_id) as affected_documents,
                            ROUND(AVG(retry_count), 2) as avg_retries
                        FROM processing_errors 
                        GROUP BY failure_scope, error_type
                        ORDER BY failure_scope, error_count DESC
                    """)
                    
                    error_distribution = [dict(row) for row in cur.fetchall()]
                    
                    return {
                        'scope_breakdown': scope_breakdown,
                        'error_distribution_by_scope': error_distribution
                    }
                    
        except Exception as e:
            logger.error(f"Error getting failure breakdown by scope: {e}")
            return {'scope_breakdown': [], 'error_distribution_by_scope': []}
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5, 
                         chunking_method: Optional[str] = None,
                         source_type: Optional[str] = None,
                         min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        """Enhanced similarity search with filtering options"""
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        query_embedding_list = query_embedding.tolist()
        
        # Build WHERE clause
        where_conditions = ["dc.embedding IS NOT NULL"]
        params = [query_embedding_list]
        param_count = 1
        
        if chunking_method:
            where_conditions.append(f"dc.chunking_method = ${param_count + 1}")
            params.append(chunking_method)
            param_count += 1
        
        if source_type:
            where_conditions.append(f"dc.source_type = ${param_count + 1}")
            params.append(source_type)
            param_count += 1
        
        where_clause = " AND ".join(where_conditions)
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT 
                        dc.id,
                        dc.content,
                        dc.chunk_index,
                        dc.chunking_method,
                        dc.similarity_score,
                        dc.llm_confidence,
                        dc.source_type,
                        dc.s3_key,
                        dc.file_path,
                        dc.metadata,
                        d.file_name,
                        d.file_type,
                        1 - (dc.embedding <=> %s::vector) AS similarity
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE {where_clause}
                    ORDER BY dc.embedding <=> %s::vector
                    LIMIT %s
                """, params + [query_embedding_list, k])
                
                results = cur.fetchall()
        
        # Filter by minimum similarity if specified
        if min_similarity > 0:
            results = [r for r in results if r['similarity'] >= min_similarity]
        
        # Convert to list of dictionaries
        return [dict(row) for row in results]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics - MCP compatible method name"""
        return self.get_document_stats()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get comprehensive document and chunking statistics"""
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                # Get overall document stats
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_docs,
                        COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_docs,
                        COUNT(CASE WHEN processing_status = 'processing' THEN 1 END) as processing_docs
                    FROM documents
                """)
                doc_stats = dict(cur.fetchone())
                
                # Get chunk stats separately
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        AVG(content_length) as avg_chunk_length
                    FROM document_chunks
                """)
                chunk_stats = dict(cur.fetchone())
                
                # Combine the stats
                overall_stats = {**doc_stats, **chunk_stats}
                
                # Get chunking method breakdown
                cur.execute("""
                    SELECT 
                        chunking_method,
                        COUNT(*) as document_count,
                        SUM(total_chunks) as total_chunks,
                        AVG(total_chunks) as avg_chunks_per_doc
                    FROM documents 
                    WHERE processing_status = 'completed'
                    GROUP BY chunking_method
                    ORDER BY document_count DESC
                """)
                chunking_stats = [dict(row) for row in cur.fetchall()]
                
                # Get source type breakdown
                cur.execute("""
                    SELECT 
                        source_type,
                        COUNT(*) as document_count,
                        SUM(total_chunks) as total_chunks
                    FROM documents 
                    GROUP BY source_type
                """)
                source_stats = [dict(row) for row in cur.fetchall()]
        
        return {
            'overall': overall_stats,
            'by_chunking_method': chunking_stats,
            'by_source_type': source_stats
        }
    
    def get_recent_documents(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently processed documents"""
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        d.*,
                        COUNT(dc.id) as actual_chunks
                    FROM documents d
                    LEFT JOIN document_chunks dc ON d.id = dc.document_id
                    GROUP BY d.id
                    ORDER BY d.created_at DESC
                    LIMIT %s
                """, (limit,))
                
                return [dict(row) for row in cur.fetchall()]
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks"""
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Delete chunks first (due to foreign key)
                    cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                    chunks_deleted = cur.rowcount
                    
                    # Delete document
                    cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
                    doc_deleted = cur.rowcount
                    
                    conn.commit()
                    
                    logger.info(f"Deleted document {document_id}: {chunks_deleted} chunks, {doc_deleted} document")
                    return doc_deleted > 0
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def search_by_content(self, 
                         search_term: str, 
                         limit: int = 10,
                         chunking_method: Optional[str] = None) -> List[Dict[str, Any]]:
        """Full-text search on chunk content"""
        
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                where_clause = "to_tsvector('english', dc.content) @@ plainto_tsquery('english', %s)"
                params = [search_term]
                
                if chunking_method:
                    where_clause += " AND dc.chunking_method = %s"
                    params.append(chunking_method)
                
                cur.execute(f"""
                    SELECT 
                        dc.content,
                        dc.chunking_method,
                        dc.chunk_index,
                        d.file_name,
                        dc.s3_key,
                        dc.file_path,
                        ts_rank(to_tsvector('english', dc.content), 
                               plainto_tsquery('english', %s)) as rank
                    FROM document_chunks dc
                    JOIN documents d ON dc.document_id = d.id
                    WHERE {where_clause}
                    ORDER BY rank DESC
                    LIMIT %s
                """, [search_term] + params + [limit])
                
                return [dict(row) for row in cur.fetchall()]

    def get_document_chunks(self, s3_key: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document by S3 key
        
        Args:
            s3_key: S3 key of the document
            
        Returns:
            List of chunk dictionaries
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            dc.id,
                            dc.content,
                            dc.embedding,
                            dc.chunking_method,
                            dc.chunk_index,
                            dc.s3_key,
                            dc.file_path,
                            dc.created_at,
                            d.file_name,
                            d.file_size,
                            d.file_type
                        FROM document_chunks dc
                        JOIN documents d ON dc.document_id = d.id
                        WHERE dc.s3_key = %s
                        ORDER BY dc.chunk_index
                    """, (s3_key,))
                    
                    return [dict(row) for row in cur.fetchall()]
                    
        except Exception as e:
            logger.error(f"Error getting document chunks for {s3_key}: {str(e)}")
            return []
    
    def delete_document_by_s3_key(self, s3_key: str) -> bool:
        """
        Delete a document and all its chunks by S3 key
        
        Args:
            s3_key: S3 key of the document to delete
            
        Returns:
            True if document was deleted, False if not found
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # First find the document ID
                    cur.execute("SELECT id FROM documents WHERE s3_key = %s", (s3_key,))
                    result = cur.fetchone()
                    
                    if not result:
                        logger.warning(f"Document with S3 key {s3_key} not found")
                        return False
                    
                    document_id = result[0]
                    
                    # Delete chunks first (due to foreign key constraint)
                    cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                    chunks_deleted = cur.rowcount
                    
                    # Delete document record
                    cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
                    document_deleted = cur.rowcount
                    
                    conn.commit()
                    
                    logger.info(f"Deleted document {s3_key}: {chunks_deleted} chunks, {document_deleted} document record")
                    return document_deleted > 0
                    
        except Exception as e:
            logger.error(f"Error deleting document {s3_key}: {str(e)}")
            return False
    
    def close_connection_pool(self):
        """Clean up connection pool on shutdown"""
        if self._connection_pool is not None:
            try:
                self._connection_pool.closeall()
                logger.info("Database connection pool closed")
            except Exception as e:
                logger.warning(f"Error closing connection pool: {e}")
    
    def __del__(self):
        """Ensure connection pool is cleaned up"""
        self.close_connection_pool()