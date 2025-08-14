"""
Error Tracking Utility for Document Processing
Provides comprehensive error logging and management for the RAG chunking system.
"""

import json
import traceback
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum

from .vector_db_helper import VectorDBHelper


class ErrorType(Enum):
    """Enumeration of error types for categorization"""
    FILE_ACCESS = "file_access"
    PASSWORD_PROTECTED = "password_protected"
    FORMAT_UNSUPPORTED = "format_unsupported"
    PARSING_ERROR = "parsing_error"
    CHUNKING_ERROR = "chunking_error"
    EMBEDDING_ERROR = "embedding_error"
    DATABASE_ERROR = "database_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    OTHER = "other"


class ProcessingStage(Enum):
    """Enumeration of processing stages where errors can occur"""
    FILE_DOWNLOAD = "file_download"
    FILE_EXTRACTION = "file_extraction"
    TEXT_CHUNKING = "text_chunking"
    EMBEDDING_GENERATION = "embedding_generation"
    DATABASE_STORAGE = "database_storage"
    VALIDATION = "validation"


class ErrorTracker:
    """
    Comprehensive error tracking system for document processing pipeline.
    Logs errors with detailed context, categorization, and recovery suggestions.
    """
    
    def __init__(self, db_helper: VectorDBHelper):
        self.db_helper = db_helper
    
    def log_error(
        self,
        document_id: Optional[uuid.UUID],
        file_name: str,
        error_type: ErrorType,
        processing_stage: ProcessingStage,
        error_message: str,
        file_path: Optional[str] = None,
        s3_key: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        is_recoverable: bool = True,
        retry_count: int = 0
    ) -> uuid.UUID:
        """
        Log a processing error with comprehensive details.
        
        Args:
            document_id: UUID of the document being processed (if available)
            file_name: Name of the file that caused the error
            error_type: Category of error (from ErrorType enum)
            processing_stage: Stage where error occurred (from ProcessingStage enum)
            error_message: Human-readable error description
            file_path: Local file path (if applicable)
            s3_key: S3 object key (if applicable)
            error_details: Additional context and metadata
            exception: Original exception object (for stack trace)
            is_recoverable: Whether this error can potentially be retried
            retry_count: Number of retry attempts already made
            
        Returns:
            UUID of the created error record
        """
        
        # Build error details with exception information
        details = error_details or {}
        if exception:
            details.update({
                'exception_type': type(exception).__name__,
                'exception_args': str(exception.args),
                'stack_trace': traceback.format_exc(),
                'timestamp': datetime.utcnow().isoformat()
            })
        
        # Store original document_id for status update (before string conversion)
        original_document_id = document_id
        
        # Convert UUID objects to strings to prevent database adapter errors
        if document_id is not None:
            document_id = str(document_id)
        
        # Insert error record
        error_id = uuid.uuid4()
        
        with self.db_helper._get_connection() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO processing_errors 
                        (id, document_id, file_name, file_path, s3_key, failure_scope, error_type, 
                         error_message, error_details, processing_stage, retry_count, 
                         is_recoverable, resolved, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        str(error_id),  # Convert error_id UUID to string as well
                        document_id,
                        file_name,
                        file_path,
                        s3_key,
                        'document',  # Document-level failure
                        error_type.value,
                        error_message,
                        json.dumps(details),
                        processing_stage.value,
                        retry_count,
                        is_recoverable,
                        False,  # resolved = False initially
                        datetime.utcnow()
                    ))
                conn.commit()
                
                # Also update document status if document_id is provided (use original UUID)
                if original_document_id:
                    self._update_document_error_status(original_document_id, error_message)
                    
            except Exception as e:
                conn.rollback()
                # If we can't log the error to database, at least log to console
                print(f"Failed to log error to database: {e}")
                print(f"Original error: {error_message}")
                raise
        
        return error_id
    
    def _update_document_error_status(self, document_id: uuid.UUID, error_message: str):
        """Update document status to failed with error message"""
        with self.db_helper._get_connection() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE documents 
                        SET processing_status = 'failed', 
                            error_message = %s,
                            updated_at = %s
                        WHERE id = %s
                    """, (error_message, datetime.utcnow(), document_id))
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Failed to update document status: {e}")
    
    def resolve_error(
        self, 
        error_id: uuid.UUID, 
        resolution_notes: str,
        resolved_by: Optional[str] = None
    ) -> bool:
        """
        Mark an error as resolved with resolution notes.
        
        Args:
            error_id: UUID of the error to resolve
            resolution_notes: Description of how the error was resolved
            resolved_by: Who/what resolved the error
            
        Returns:
            True if successfully resolved, False otherwise
        """
        with self.db_helper._get_connection() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE processing_errors 
                        SET resolved = true,
                            resolution_notes = %s,
                            resolved_at = %s,
                            updated_at = %s
                        WHERE id = %s
                    """, (
                        f"{resolution_notes} (Resolved by: {resolved_by or 'system'})",
                        datetime.utcnow(),
                        datetime.utcnow(),
                        error_id
                    ))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                conn.rollback()
                print(f"Failed to resolve error {error_id}: {e}")
                return False
    
    def increment_retry_count(self, error_id: uuid.UUID) -> bool:
        """Increment the retry count for an error"""
        with self.db_helper._get_connection() as conn:
            conn.autocommit = False
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        UPDATE processing_errors 
                        SET retry_count = retry_count + 1,
                            updated_at = %s
                        WHERE id = %s
                    """, (datetime.utcnow(), error_id))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                conn.rollback()
                print(f"Failed to increment retry count for error {error_id}: {e}")
                return False
    
    def get_errors_by_document(self, document_id: uuid.UUID) -> List[Dict[str, Any]]:
        """Get all errors for a specific document"""
        with self.db_helper._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, error_type, error_message, error_details, 
                           processing_stage, retry_count, is_recoverable, 
                           resolved, resolution_notes, created_at, resolved_at
                    FROM processing_errors 
                    WHERE document_id = %s
                    ORDER BY created_at DESC
                """, (document_id,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all errors"""
        with self.db_helper._get_connection() as conn:
            with conn.cursor() as cursor:
                # Get error counts by type
                cursor.execute("""
                    SELECT error_type, COUNT(*) as count,
                           COUNT(CASE WHEN resolved = true THEN 1 END) as resolved_count
                    FROM processing_errors 
                    GROUP BY error_type
                    ORDER BY count DESC
                """)
                error_types = [
                    {'type': row[0], 'total': row[1], 'resolved': row[2], 'unresolved': row[1] - row[2]}
                    for row in cursor.fetchall()
                ]
                
                # Get error counts by stage
                cursor.execute("""
                    SELECT processing_stage, COUNT(*) as count,
                           COUNT(CASE WHEN resolved = true THEN 1 END) as resolved_count
                    FROM processing_errors 
                    GROUP BY processing_stage
                    ORDER BY count DESC
                """)
                stages = [
                    {'stage': row[0], 'total': row[1], 'resolved': row[2], 'unresolved': row[1] - row[2]}
                    for row in cursor.fetchall()
                ]
                
                # Get recent unresolved errors
                cursor.execute("""
                    SELECT file_name, error_type, processing_stage, error_message, created_at
                    FROM processing_errors 
                    WHERE resolved = false
                    ORDER BY created_at DESC
                    LIMIT 10
                """)
                columns = [desc[0] for desc in cursor.description]
                recent_errors = [dict(zip(columns, row)) for row in cursor.fetchall()]
                
                return {
                    'error_types': error_types,
                    'processing_stages': stages,
                    'recent_unresolved': recent_errors
                }
    
    def get_password_protected_files(self) -> List[Dict[str, Any]]:
        """Get all password-protected file errors"""
        with self.db_helper._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT file_name, file_path, s3_key, error_message, 
                           error_details, created_at, retry_count
                    FROM processing_errors 
                    WHERE error_type = %s
                    ORDER BY created_at DESC
                """, (ErrorType.PASSWORD_PROTECTED.value,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_recoverable_errors(self, max_retries: int = 3) -> List[Dict[str, Any]]:
        """Get errors that can be retried (recoverable and under retry limit)"""
        with self.db_helper._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, document_id, file_name, error_type, processing_stage,
                           error_message, retry_count, created_at
                    FROM processing_errors 
                    WHERE is_recoverable = true 
                      AND resolved = false 
                      AND retry_count < %s
                    ORDER BY created_at ASC
                """, (max_retries,))
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]


# Convenience functions for common error scenarios
def log_password_protected_error(
    error_tracker: ErrorTracker,
    document_id: Optional[uuid.UUID],
    file_name: str,
    file_path: Optional[str] = None,
    s3_key: Optional[str] = None
) -> uuid.UUID:
    """Log a password-protected file error"""
    return error_tracker.log_error(
        document_id=document_id,
        file_name=file_name,
        error_type=ErrorType.PASSWORD_PROTECTED,
        processing_stage=ProcessingStage.FILE_EXTRACTION,
        error_message=f"File '{file_name}' is password protected and cannot be processed",
        file_path=file_path,
        s3_key=s3_key,
        error_details={
            'suggested_action': 'Provide password or process manually',
            'is_security_related': True
        },
        is_recoverable=False  # Cannot automatically recover from password protection
    )


def log_unsupported_format_error(
    error_tracker: ErrorTracker,
    document_id: Optional[uuid.UUID],
    file_name: str,
    file_extension: str,
    file_path: Optional[str] = None,
    s3_key: Optional[str] = None
) -> uuid.UUID:
    """Log an unsupported file format error"""
    return error_tracker.log_error(
        document_id=document_id,
        file_name=file_name,
        error_type=ErrorType.FORMAT_UNSUPPORTED,
        processing_stage=ProcessingStage.FILE_EXTRACTION,
        error_message=f"File format '{file_extension}' is not supported for processing",
        file_path=file_path,
        s3_key=s3_key,
        error_details={
            'file_extension': file_extension,
            'suggested_action': 'Convert to supported format or add format support'
        },
        is_recoverable=True  # Can be recovered by adding format support
    )


def log_chunking_error(
    error_tracker: ErrorTracker,
    document_id: Optional[uuid.UUID],
    file_name: str,
    exception: Exception,
    chunking_method: str,
    file_path: Optional[str] = None,
    s3_key: Optional[str] = None
) -> uuid.UUID:
    """Log a chunking process error"""
    return error_tracker.log_error(
        document_id=document_id,
        file_name=file_name,
        error_type=ErrorType.CHUNKING_ERROR,
        processing_stage=ProcessingStage.TEXT_CHUNKING,
        error_message=f"Failed to chunk document using {chunking_method} method: {str(exception)}",
        file_path=file_path,
        s3_key=s3_key,
        error_details={
            'chunking_method': chunking_method,
            'suggested_action': 'Try different chunking method or check document content'
        },
        exception=exception,
        is_recoverable=True
    )
