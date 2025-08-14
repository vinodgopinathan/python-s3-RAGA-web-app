import os
import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBHelper:
    def __init__(self):
        self.host = os.environ.get('POSTGRES_HOST', 'localhost')
        self.port = os.environ.get('POSTGRES_PORT', '5432')
        self.user = os.environ.get('POSTGRES_USER', 'postgres')
        self.password = os.environ.get('POSTGRES_PASSWORD')
        self.database = os.environ.get('POSTGRES_DB', 'postgres')
        
        # Initialize embedding model
        self.embedding_model_name = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize database connection
        self._init_database()
    
    def _get_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return conn
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def _init_database(self):
        """Initialize database with pgvector extension and create tables if not exist"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Enable pgvector extension
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table if it doesn't exist
            cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                file_path TEXT NOT NULL,
                s3_key TEXT,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size BIGINT,
                content_type TEXT,
                source_type TEXT NOT NULL CHECK (source_type IN ('local', 's3')),
                chunking_method TEXT NOT NULL,
                chunk_size INTEGER DEFAULT 1000,
                chunk_overlap INTEGER DEFAULT 200,
                total_chunks INTEGER DEFAULT 0,
                processing_status TEXT DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP WITH TIME ZONE
            );
            """)
            
            # Create document_chunks table if it doesn't exist
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
                s3_key TEXT,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_length INTEGER,
                metadata JSONB DEFAULT '{{}}',
                embedding vector({self.embedding_dimension}),
                chunking_method TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(document_id, chunk_index),
                UNIQUE(s3_key, chunk_index)
            );
            """
            
            cursor.execute(create_table_query)
            
            # Add unique constraints if they don't exist (for existing tables)
            cursor.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint 
                        WHERE conname = 'unique_s3_key_chunk_index'
                    ) THEN
                        ALTER TABLE document_chunks 
                        ADD CONSTRAINT unique_s3_key_chunk_index 
                        UNIQUE (s3_key, chunk_index);
                    END IF;
                END $$;
            """)
            
            cursor.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint 
                        WHERE conname = 'unique_document_chunk_index'
                    ) THEN
                        ALTER TABLE document_chunks 
                        ADD CONSTRAINT unique_document_chunk_index 
                        UNIQUE (document_id, chunk_index);
                    END IF;
                END $$;
            """)
            
            # Create indexes for better performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_s3_key 
                ON document_chunks(s3_key);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id 
                ON document_chunks(document_id);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
                ON document_chunks USING ivfflat (embedding vector_cosine_ops);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_processing_status 
                ON documents(processing_status);
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        try:
            embedding = self.embedding_model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def store_document_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
        """Store document chunks with embeddings in the database
        
        Args:
            document_id: UUID of the document these chunks belong to
            chunks: List of chunk dictionaries with content and metadata
            
        Returns:
            Tuple of (stored_count, failed_chunks_list)
        """
        if not chunks:
            logger.warning(f"No chunks provided for document {document_id}")
            return 0, []
            
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Start transaction
            conn.autocommit = False
            
            # For backward compatibility, also support s3_key-based storage
            s3_key = None
            if chunks and 'metadata' in chunks[0]:
                s3_key = chunks[0]['metadata'].get('s3_key')
            
            # Delete existing chunks for this document to prevent duplicates
            if document_id:
                cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} existing chunks for document {document_id}")
            elif s3_key:
                cursor.execute("DELETE FROM document_chunks WHERE s3_key = %s", (s3_key,))
                deleted_count = cursor.rowcount
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} existing chunks for S3 key {s3_key}")
            
            # Validate and fix chunk indices to prevent duplicates
            validated_chunks = self._validate_and_fix_chunk_indices(chunks)
            
            stored_count = 0
            failed_chunks = []
            
            for chunk in validated_chunks:
                try:
                    content = chunk['content']
                    chunk_index = chunk.get('chunk_index', 0)
                    metadata = chunk.get('metadata', {})
                    
                    # Generate embedding
                    embedding = self.generate_embedding(content)
                    
                    # Convert embedding to string format for PostgreSQL vector type
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Extract metadata fields
                    chunking_method = metadata.get('chunking_method', 'unknown')
                    content_length = len(content)
                    chunk_s3_key = metadata.get('s3_key', s3_key)
                    source_type = metadata.get('source_type', 's3')  # Default to 's3' for S3 files
                    
                    # Use ON CONFLICT to handle any remaining duplicate issues
                    insert_query = """
                    INSERT INTO document_chunks 
                    (document_id, s3_key, chunk_index, content, content_length, embedding, 
                     chunking_method, source_type, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s, %s, %s)
                    ON CONFLICT (document_id, chunk_index) 
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        content_length = EXCLUDED.content_length,
                        embedding = EXCLUDED.embedding,
                        chunking_method = EXCLUDED.chunking_method,
                        source_type = EXCLUDED.source_type,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.created_at
                    """
                    
                    cursor.execute(insert_query, (
                        document_id,
                        chunk_s3_key,
                        chunk_index,
                        content,
                        content_length,
                        embedding_str,
                        chunking_method,
                        source_type,
                        json.dumps(metadata),
                        datetime.utcnow()
                    ))
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Error storing chunk {chunk_index} for document {document_id}: {str(e)}")
                    # Add error metadata for proper error handling
                    failed_chunk = chunk.copy()
                    failed_chunk.update({
                        'error_type': 'database_error',
                        'error_message': str(e),
                        'content_preview': content[:100] if 'content' in locals() else 'No content'
                    })
                    failed_chunks.append(failed_chunk)
                    # Continue processing other chunks
                    continue
            
            # Commit transaction
            conn.commit()
            conn.autocommit = True
            
            if failed_chunks:
                logger.warning(f"Stored {stored_count}/{len(validated_chunks)} chunks for document: {document_id} ({len(failed_chunks)} failed)")
            else:
                logger.info(f"Successfully stored {stored_count} chunks for document: {document_id}")
            
            return stored_count, failed_chunks
            
        except Exception as e:
            logger.error(f"Error storing document chunks: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
                conn.autocommit = True
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _validate_and_fix_chunk_indices(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix chunk indices to prevent duplicates"""
        if not chunks:
            return chunks
        
        # Check if chunk indices are properly set
        seen_indices = set()
        fixed_chunks = []
        next_index = 0
        
        for i, chunk in enumerate(chunks):
            original_index = chunk.get('chunk_index', i)
            
            # If index is already seen or not set properly, assign sequential index
            if original_index in seen_indices or original_index is None:
                while next_index in seen_indices:
                    next_index += 1
                chunk_index = next_index
                logger.warning(f"Fixed duplicate/missing chunk index {original_index} -> {chunk_index}")
            else:
                chunk_index = original_index
            
            # Update chunk with validated index
            validated_chunk = chunk.copy()
            validated_chunk['chunk_index'] = chunk_index
            fixed_chunks.append(validated_chunk)
            
            seen_indices.add(chunk_index)
            next_index = max(next_index, chunk_index + 1)
        
        return fixed_chunks
    
    def fix_duplicate_chunks(self) -> Dict[str, int]:
        """Fix existing duplicate chunks in the database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Find documents with duplicate chunk indices
            cursor.execute("""
                SELECT s3_key, chunk_index, COUNT(*) as duplicate_count
                FROM document_chunks
                GROUP BY s3_key, chunk_index
                HAVING COUNT(*) > 1
                ORDER BY s3_key, chunk_index
            """)
            
            duplicates = cursor.fetchall()
            
            if not duplicates:
                logger.info("No duplicate chunks found")
                return {"documents_fixed": 0, "duplicates_removed": 0}
            
            documents_fixed = set()
            total_duplicates_removed = 0
            
            for s3_key, chunk_index, duplicate_count in duplicates:
                logger.warning(f"Found {duplicate_count} duplicates for {s3_key} chunk {chunk_index}")
                
                # Keep only the first occurrence, delete the rest
                cursor.execute("""
                    DELETE FROM document_chunks 
                    WHERE id IN (
                        SELECT id FROM document_chunks 
                        WHERE s3_key = %s AND chunk_index = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    )
                """, (s3_key, chunk_index, duplicate_count - 1))
                
                removed_count = cursor.rowcount
                total_duplicates_removed += removed_count
                documents_fixed.add(s3_key)
                
                logger.info(f"Removed {removed_count} duplicate chunks for {s3_key} chunk {chunk_index}")
            
            conn.commit()
            
            result = {
                "documents_fixed": len(documents_fixed),
                "duplicates_removed": total_duplicates_removed
            }
            
            logger.info(f"Fixed duplicates: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error fixing duplicate chunks: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def find_existing_document(self, s3_key: str, file_name: str) -> Optional[str]:
        """Find existing document by S3 key or file name
        
        Returns:
            Document ID if found, None otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # First try to find by s3_key (most reliable)
            if s3_key:
                cursor.execute("SELECT id FROM documents WHERE s3_key = %s", (s3_key,))
                result = cursor.fetchone()
                if result:
                    logger.info(f"Found existing document by S3 key: {s3_key}")
                    return result[0]
            
            # If not found by s3_key, try by file_name
            cursor.execute("SELECT id FROM documents WHERE file_name = %s", (file_name,))
            result = cursor.fetchone()
            if result:
                logger.info(f"Found existing document by file name: {file_name}")
                return result[0]
                
            return None
            
        except Exception as e:
            logger.error(f"Error finding existing document: {str(e)}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def delete_document_and_chunks(self, document_id: str) -> bool:
        """Delete a document and all its chunks
        
        Args:
            document_id: UUID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Delete chunks first (due to foreign key constraint)
            cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
            chunks_deleted = cursor.rowcount
            
            # Delete document
            cursor.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            document_deleted = cursor.rowcount
            
            conn.commit()
            
            if document_deleted > 0:
                logger.info(f"Deleted document {document_id} and {chunks_deleted} chunks")
                return True
            else:
                logger.warning(f"Document {document_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document and chunks: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
            return False
        finally:
            if 'conn' in locals():
                conn.close()

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
        """Create a new document record and return document ID
        
        If a document with the same S3 key already exists, it will be deleted
        and replaced with the new one to prevent duplicates.
        """
        
        # Check if document already exists
        existing_doc_id = self.find_existing_document(s3_key, file_name)
        if existing_doc_id:
            logger.info(f"Document already exists (ID: {existing_doc_id}), replacing it")
            self.delete_document_and_chunks(existing_doc_id)
        
        document_id = str(uuid.uuid4())
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
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
            
        except Exception as e:
            logger.error(f"Error creating document record: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def update_document_status(self, 
                             document_id: str, 
                             status: str, 
                             total_chunks: Optional[int] = None,
                             error_message: Optional[str] = None):
        """Update document processing status"""
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if status == 'completed':
                cursor.execute("""
                    UPDATE documents 
                    SET processing_status = %s, total_chunks = %s, 
                        processed_at = %s, updated_at = %s
                    WHERE id = %s
                """, (status, total_chunks, datetime.utcnow(), datetime.utcnow(), document_id))
            elif status == 'failed':
                cursor.execute("""
                    UPDATE documents 
                    SET processing_status = %s, error_message = %s, updated_at = %s
                    WHERE id = %s
                """, (status, error_message, datetime.utcnow(), document_id))
            else:
                cursor.execute("""
                    UPDATE documents 
                    SET processing_status = %s, updated_at = %s
                    WHERE id = %s
                """, (status, datetime.utcnow(), document_id))
            
            conn.commit()
            logger.info(f"Updated document {document_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Error updating document status: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def store_chunks_legacy(self, s3_key: str, chunks: List[Dict[str, Any]]) -> bool:
        """Legacy method for backward compatibility - store chunks by s3_key only"""
        try:
            # Create a dummy document record for legacy calls
            document_id = str(uuid.uuid4())
            stored_count, failed_chunks = self.store_document_chunks(document_id, chunks)
            return stored_count > 0
        except Exception as e:
            logger.error(f"Error in legacy store_chunks: {str(e)}")
            return False
    
    def store_document_chunks_with_metadata(self, document_id: str, chunks: List[Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
        """Store document chunks with full metadata support and improved duplicate handling"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Delete existing chunks for this document
            cursor.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
            
            # Validate and fix chunk indices to prevent duplicates
            validated_chunks = self._validate_and_fix_chunk_indices(chunks)
            
            stored_count = 0
            failed_chunks = []
            
            for chunk in validated_chunks:
                try:
                    content = chunk['content']
                    chunk_index = chunk.get('chunk_index', 0)
                    metadata = chunk.get('metadata', {})
                    
                    # Generate embedding
                    embedding = self.generate_embedding(content)
                    
                    # Convert embedding to string format for PostgreSQL vector type
                    embedding_str = '[' + ','.join(map(str, embedding)) + ']'
                    
                    # Extract metadata fields
                    chunking_method = metadata.get('chunking_method', 'unknown')
                    content_length = len(content)
                    source_type = metadata.get('source_type', 's3')  # Default to 's3' for S3 files
                    
                    insert_query = """
                    INSERT INTO document_chunks 
                    (document_id, chunk_index, content, content_length, embedding, 
                     chunking_method, source_type, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s::vector, %s, %s, %s, %s)
                    """
                    
                    cursor.execute(insert_query, (
                        document_id,
                        chunk_index,
                        content,
                        content_length,
                        embedding_str,
                        chunking_method,
                        source_type,
                        json.dumps(metadata),
                        datetime.utcnow()
                    ))
                    stored_count += 1
                    
                except psycopg2.IntegrityError as e:
                    logger.warning(f"Duplicate chunk index {chunk_index} for document {document_id}, skipping")
                    # Add error metadata for proper error handling
                    failed_chunk = chunk.copy()
                    failed_chunk.update({
                        'error_type': 'database_error',
                        'error_message': f"Integrity constraint violation: {str(e)}",
                        'content_preview': content[:100] if content else 'No content'
                    })
                    failed_chunks.append(failed_chunk)
                    continue
                except Exception as e:
                    logger.error(f"Error storing chunk {chunk_index} for document {document_id}: {str(e)}")
                    # Add error metadata for proper error handling
                    failed_chunk = chunk.copy()
                    failed_chunk.update({
                        'error_type': 'database_error',
                        'error_message': str(e),
                        'content_preview': content[:100] if content else 'No content'
                    })
                    failed_chunks.append(failed_chunk)
                    continue
            
            conn.commit()
            
            if failed_chunks:
                logger.warning(f"Stored {stored_count}/{len(validated_chunks)} chunks for document: {document_id} ({len(failed_chunks)} failed)")
            else:
                logger.info(f"Stored {stored_count} chunks for document: {document_id}")
            
            return stored_count, failed_chunks
            
        except Exception as e:
            logger.error(f"Error storing document chunks: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def similarity_search(self, query: str, limit: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform similarity search for relevant document chunks"""
        try:
            # Generate embedding for query
            query_embedding = self.generate_embedding(query)
            
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Convert embedding list to a format PostgreSQL can handle
            # Format as a string representation of a vector
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Perform similarity search using cosine similarity with proper join
            search_query = """
            SELECT 
                dc.id,
                COALESCE(dc.s3_key, d.s3_key) as s3_key,
                dc.chunk_index,
                dc.content,
                dc.metadata,
                dc.chunking_method,
                1 - (dc.embedding <=> %s::vector) as similarity_score
            FROM document_chunks dc
            LEFT JOIN documents d ON dc.document_id = d.id
            WHERE dc.embedding IS NOT NULL
            AND 1 - (dc.embedding <=> %s::vector) > %s
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s;
            """
            
            cursor.execute(search_query, (
                embedding_str,
                embedding_str,
                similarity_threshold,
                embedding_str,
                limit
            ))
            
            results = cursor.fetchall()
            
            # Convert to list of dictionaries
            search_results = []
            for row in results:
                # Handle metadata - it might be a dict (JSONB) or string (JSON)
                metadata = row['metadata']
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                elif metadata is None:
                    metadata = {}
                # If it's already a dict, use it as-is
                
                result = {
                    'id': row['id'],
                    's3_key': row['s3_key'],
                    'chunk_index': row['chunk_index'],
                    'content': row['content'],
                    'metadata': metadata,
                    'chunking_method': row['chunking_method'],
                    'similarity_score': float(row['similarity_score'])
                }
                search_results.append(result)
            
            logger.info(f"Found {len(search_results)} similar chunks for query: '{query[:50]}...'")
            return search_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_document_chunks(self, s3_key: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT id, s3_key, chunk_index, content, metadata, created_at
            FROM document_chunks
            WHERE s3_key = %s
            ORDER BY chunk_index;
            """
            
            cursor.execute(query, (s3_key,))
            results = cursor.fetchall()
            
            chunks = []
            for row in results:
                chunk = {
                    'id': row['id'],
                    's3_key': row['s3_key'],
                    'chunk_index': row['chunk_index'],
                    'content': row['content'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {},
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def delete_document(self, s3_key: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM document_chunks WHERE s3_key = %s", (s3_key,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            logger.info(f"Deleted {deleted_count} chunks for document: {s3_key}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            if 'conn' in locals():
                conn.rollback()
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_document_count(self) -> int:
        """Get total number of unique documents in the database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(DISTINCT s3_key) FROM document_chunks")
            count = cursor.fetchone()[0]
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get comprehensive document and chunking statistics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get overall document stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_docs,
                    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_docs,
                    COUNT(CASE WHEN processing_status = 'processing' THEN 1 END) as processing_docs
                FROM documents
            """)
            doc_stats_row = cursor.fetchone()
            doc_stats = dict(doc_stats_row) if doc_stats_row else {}
            
            # Get chunk stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    AVG(content_length) as avg_chunk_length
                FROM document_chunks
            """)
            chunk_stats_row = cursor.fetchone()
            chunk_stats = dict(chunk_stats_row) if chunk_stats_row else {}
            
            # Combine the stats
            overall_stats = {**doc_stats, **chunk_stats}
            
            # Get chunking method breakdown
            cursor.execute("""
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
            chunking_stats = [dict(row) for row in cursor.fetchall()]
            
            # Get source type breakdown
            cursor.execute("""
                SELECT 
                    source_type,
                    COUNT(*) as document_count,
                    SUM(total_chunks) as total_chunks
                FROM documents 
                GROUP BY source_type
            """)
            source_stats = [dict(row) for row in cursor.fetchall()]
            
            return {
                'overall': overall_stats,
                'by_chunking_method': chunking_stats,
                'by_source_type': source_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting document stats: {str(e)}")
            # Return default stats on error
            return {
                'overall': {'total_documents': 0, 'completed_docs': 0, 'failed_docs': 0, 'processing_docs': 0, 'total_chunks': 0, 'avg_chunk_length': 0},
                'by_chunking_method': [],
                'by_source_type': []
            }
        finally:
            if 'conn' in locals():
                conn.close()
    
        
    def get_documents(self, limit: int = 50, status: str = None) -> List[Dict[str, Any]]:
        """Get list of documents with their metadata"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if status:
                query = """
                SELECT id, s3_key, filename, content_type, file_size, 
                       chunk_count, status, created_at, updated_at, metadata
                FROM documents 
                WHERE status = %s
                ORDER BY updated_at DESC 
                LIMIT %s
                """
                cursor.execute(query, (status, limit))
            else:
                query = """
                SELECT id, s3_key, filename, content_type, file_size, 
                       chunk_count, status, created_at, updated_at, metadata
                FROM documents 
                ORDER BY updated_at DESC 
                LIMIT %s
                """
                cursor.execute(query, (limit,))
            
            documents = []
            for row in cursor.fetchall():
                doc = {
                    'id': row[0],
                    's3_key': row[1],
                    'filename': row[2],
                    'content_type': row[3],
                    'file_size': row[4],
                    'chunk_count': row[5],
                    'status': row[6],
                    'created_at': row[7].isoformat() if row[7] else None,
                    'updated_at': row[8].isoformat() if row[8] else None,
                    'metadata': row[9] if row[9] else {}
                }
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_recent_documents(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently processed documents"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT 
                    d.*,
                    COUNT(dc.id) as actual_chunks
                FROM documents d
                LEFT JOIN document_chunks dc ON d.id = dc.document_id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                LIMIT %s
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
            
        except Exception as e:
            logger.error(f"Error getting recent documents: {str(e)}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks in the database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM document_chunks")
            count = cursor.fetchone()[0]
            
            return count
            
        except Exception as e:
            logger.error(f"Error getting chunk count: {str(e)}")
            raise
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive vector database statistics for RAGA"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get document statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_documents,
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_documents,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_documents,
                    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_documents
                FROM document_metadata
            """)
            doc_stats = cursor.fetchone()
            
            # Get chunk statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    AVG(CASE WHEN chunk_length IS NOT NULL THEN chunk_length ELSE LENGTH(content) END) as avg_chunk_length,
                    COUNT(DISTINCT s3_key) as indexed_files,
                    COUNT(DISTINCT chunking_method) as chunking_methods_used
                FROM document_chunks
            """)
            chunk_stats = cursor.fetchone()
            
            # Get chunking method distribution
            cursor.execute("""
                SELECT 
                    chunking_method, 
                    COUNT(*) as count
                FROM document_chunks 
                WHERE chunking_method IS NOT NULL
                GROUP BY chunking_method
                ORDER BY count DESC
            """)
            chunking_methods = dict(cursor.fetchall())
            
            return {
                'total_documents': doc_stats[0] if doc_stats[0] else 0,
                'completed_documents': doc_stats[1] if doc_stats[1] else 0,
                'failed_documents': doc_stats[2] if doc_stats[2] else 0,
                'processing_documents': doc_stats[3] if doc_stats[3] else 0,
                'total_chunks': chunk_stats[0] if chunk_stats[0] else 0,
                'avg_chunk_length': float(chunk_stats[1]) if chunk_stats[1] else 0.0,
                'indexed_files': chunk_stats[2] if chunk_stats[2] else 0,
                'chunking_methods_used': chunk_stats[3] if chunk_stats[3] else 0,
                'chunking_method_distribution': chunking_methods
            }
            
        except Exception as e:
            logger.error(f"Error getting vector database stats: {str(e)}")
            return {
                'error': str(e),
                'total_documents': 0,
                'total_chunks': 0,
                'indexed_files': 0
            }
        finally:
            if 'conn' in locals():
                conn.close()
