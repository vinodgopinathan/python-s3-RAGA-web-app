import os
import logging
import boto3
from pathlib import Path
from typing import List, Dict, Any, Optional
import mimetypes
from werkzeug.utils import secure_filename
import threading
import time

# Local imports
from .vector_db_helper import VectorDBHelper
from .document_processor import DocumentProcessor
from .s3_helper import S3Helper
from .error_tracker import ErrorTracker, ErrorType, ProcessingStage, log_password_protected_error, log_unsupported_format_error, log_chunking_error

logger = logging.getLogger(__name__)

class ChunkingService:
    def __init__(self):
        """Initialize the chunking service with database and processing components"""
        self.db_helper = VectorDBHelper()
        self.document_processor = DocumentProcessor()
        self.s3_helper = S3Helper()
        self.error_tracker = ErrorTracker(self.db_helper)
        
        # Supported file extensions - now includes Excel and CSV
        self.supported_extensions = {
            '.txt', '.md', '.markdown', '.pdf', '.json', 
            '.csv', '.tsv', '.log', '.py', '.js', '.html', 
            '.xml', '.yml', '.yaml', '.rtf', '.docx',
            '.xls', '.xlsx'
        }
        
        # Processing status tracking
        self.processing_jobs = {}
        
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the current status of a processing job"""
        if job_id not in self.processing_jobs:
            return {'error': 'Job not found', 'status': 'not_found'}
        
        job = self.processing_jobs[job_id]
        
        # Calculate progress percentage
        if job.get('files_found', 0) > 0:
            progress_percentage = (job.get('files_processed', 0) / job['files_found']) * 100
        else:
            progress_percentage = 0
        
        return {
            'job_id': job_id,
            'status': job['status'],
            'total_files': job.get('files_found', 0),
            'processed_files': job.get('files_processed', 0),
            'failed_files': job.get('files_failed', 0),
            'current_file': job.get('current_file', ''),
            'progress_percentage': round(progress_percentage, 1),
            'start_time': job['start_time'],
            'end_time': job.get('end_time'),
            'duration': job.get('duration'),
            'errors': job.get('errors', []),
            'documents': job.get('documents', []),
            'directory_path': job.get('directory_path', ''),
            'chunking_method': job.get('chunking_method', ''),
            'chunk_size': job.get('chunk_size', 0),
            'chunk_overlap': job.get('chunk_overlap', 0)
        }
    
    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get status of all processing jobs"""
        return [self.get_job_status(job_id) for job_id in self.processing_jobs.keys()]
    
    def get_all_job_statuses(self) -> List[Dict[str, Any]]:
        """Get status of all processing jobs (alias for compatibility)"""
        return self.get_all_jobs()
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running processing job"""
        if job_id not in self.processing_jobs:
            return {'error': 'Job not found', 'status': 'not_found'}
        
        job = self.processing_jobs[job_id]
        if job['status'] in ['processing', 'started']:
            job['status'] = 'cancelled'
            job['end_time'] = time.time()
            job['duration'] = job['end_time'] - job['start_time']
            return {'message': 'Job cancelled successfully', 'status': 'cancelled'}
        else:
            return {'message': 'Job cannot be cancelled', 'status': job['status']}
    
    def process_directory(self, 
                         directory_path: str, 
                         chunking_method: str,
                         chunk_size: int = 1000,
                         chunk_overlap: int = 200,
                         recursive: bool = True,
                         source_type: str = 'local') -> Dict[str, Any]:
        """
        Process all files in a directory with the specified chunking method
        Can process either local directories or S3 prefixes (directories)
        Returns job information for tracking progress
        """
        
        # Generate job ID
        job_id = f"dir_{int(time.time())}_{hash(directory_path) % 10000}"
        
        # Find all supported files (local or S3)
        files_to_process = self._find_files_in_directory(directory_path, recursive, source_type)
        
        if not files_to_process:
            return {
                'job_id': job_id,
                'status': 'completed',
                'message': 'No supported files found in directory',
                'files_found': 0,
                'files_processed': 0,
                'files_failed': 0
            }
        
        # Initialize job status
        job_status = {
            'job_id': job_id,
            'status': 'processing',
            'directory_path': directory_path,
            'chunking_method': chunking_method,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'files_found': len(files_to_process),
            'files_processed': 0,
            'files_failed': 0,
            'current_file': None,
            'start_time': time.time(),
            'documents': [],
            'errors': []
        }
        
        self.processing_jobs[job_id] = job_status
        
        # Start processing in background thread
        thread = threading.Thread(
            target=self._process_directory_files,
            args=(job_id, files_to_process, chunking_method, chunk_size, chunk_overlap, source_type)
        )
        thread.daemon = True
        thread.start()
        
        return {
            'job_id': job_id,
            'status': 'started',
            'message': f'Started processing {len(files_to_process)} files',
            'files_found': len(files_to_process)
        }
    
    def process_single_file(self,
                           file_path: str = None,
                           s3_key: str = None,
                           chunking_method: str = 'adaptive',
                           chunk_size: int = 1000,
                           chunk_overlap: int = 200,
                           source_type: str = 'local') -> Dict[str, Any]:
        """
        Process a single file with the specified chunking method
        Can process either a local file or an S3 file
        """
        
        if s3_key:
            # Process S3 file
            return self._process_s3_file(
                s3_key=s3_key,
                source_type=source_type,
                chunking_method=chunking_method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif file_path:
            # Process local file
            return self._process_single_file(
                file_path=file_path,
                source_type=source_type,
                chunking_method=chunking_method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError("Either file_path or s3_key must be provided")
    
    def _find_files_in_directory(self, directory_path: str, recursive: bool = True, source_type: str = 'local') -> List[str]:
        """Find all supported files in directory (local or S3)"""
        files = []
        
        logger.info(f"Finding files in directory: {directory_path}, source_type: {source_type}, recursive: {recursive}")
        
        if source_type == 's3':
            # Handle S3 "directory" (prefix)
            try:
                # Use the S3Helper to list objects with the prefix
                logger.info(f"Calling S3Helper.list_objects with prefix: {directory_path}")
                s3_objects = self.s3_helper.list_objects(prefix=directory_path)
                logger.info(f"S3Helper returned {len(s3_objects)} objects")
                
                for obj_key in s3_objects:
                    logger.info(f"Processing S3 object: {obj_key}")
                    # Get file extension
                    file_ext = Path(obj_key).suffix.lower()
                    logger.info(f"File extension: {file_ext}")
                    
                    # Check if it's a supported file type
                    if file_ext in self.supported_extensions:
                        logger.info(f"File extension {file_ext} is supported")
                        # If not recursive, only include files directly in the prefix
                        if not recursive:
                            # Check if file is directly in the prefix (no additional subdirectories)
                            relative_path = obj_key[len(directory_path):].lstrip('/')
                            if '/' not in relative_path:
                                files.append(obj_key)
                                logger.info(f"Added non-recursive file: {obj_key}")
                        else:
                            files.append(obj_key)
                            logger.info(f"Added recursive file: {obj_key}")
                    else:
                        logger.info(f"File extension {file_ext} is not supported. Supported: {self.supported_extensions}")
                            
            except Exception as e:
                logger.error(f"Error listing S3 objects in {directory_path}: {e}")
                raise
        else:
            # Handle local directory
            path = Path(directory_path)
            
            if recursive:
                pattern = "**/*"
            else:
                pattern = "*"
            
            for file_path in path.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    files.append(str(file_path))
        
        logger.info(f"Found {len(files)} supported files")
        return files
    
    def _process_directory_files(self, 
                                job_id: str, 
                                files: List[str], 
                                chunking_method: str,
                                chunk_size: int,
                                chunk_overlap: int,
                                source_type: str):
        """Process files in background thread with detailed progress tracking"""
        
        job_status = self.processing_jobs[job_id]
        job_status['status'] = 'processing'
        
        try:
            logger.info(f"Starting to process {len(files)} files for job {job_id}")
            
            for i, file_path in enumerate(files):
                # Check if job was cancelled
                if job_status['status'] == 'cancelled':
                    logger.info(f"Job {job_id} was cancelled, stopping processing")
                    break
                
                # Update current file being processed
                job_status['current_file'] = file_path
                file_name = Path(file_path).name
                logger.info(f"Processing file {i+1}/{len(files)}: {file_name}")
                
                try:
                    # Process single file based on source type
                    if source_type == 's3':
                        result = self._process_s3_file(
                            s3_key=file_path,  # For S3, file_path is actually the S3 key
                            source_type=source_type,
                            chunking_method=chunking_method,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    else:
                        result = self._process_single_file(
                            file_path=file_path,
                            source_type=source_type,
                            chunking_method=chunking_method,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    
                    # Update progress
                    job_status['files_processed'] += 1
                    job_status['documents'].append({
                        'file_path': file_path,
                        'document_id': result['document_id'],
                        'chunks_created': result['chunks_created'],
                        'status': 'success'
                    })
                    
                    # Calculate and log progress
                    progress = (job_status['files_processed'] / len(files)) * 100
                    logger.info(f"Successfully processed {file_name}: {result['chunks_created']} chunks created")
                    logger.info(f"Progress: {job_status['files_processed']}/{len(files)} files ({progress:.1f}%)")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    job_status['files_failed'] += 1
                    
                    # Enhanced error tracking with categorization
                    document_id = None
                    file_name = os.path.basename(file_path)
                    file_ext = Path(file_path).suffix.lower()
                    
                    try:
                        # Try to categorize the error type
                        error_message = str(e).lower()
                        
                        if 'password' in error_message or 'encrypted' in error_message:
                            # Password-protected file
                            error_id = log_password_protected_error(
                                self.error_tracker, document_id, file_name, 
                                file_path=file_path if not file_path.startswith('s3://') else None,
                                s3_key=file_path if file_path.startswith('s3://') else None
                            )
                            error_type = "password_protected"
                            
                        elif file_ext not in self.supported_extensions:
                            # Unsupported format
                            error_id = log_unsupported_format_error(
                                self.error_tracker, document_id, file_name, file_ext,
                                file_path=file_path if not file_path.startswith('s3://') else None,
                                s3_key=file_path if file_path.startswith('s3://') else None
                            )
                            error_type = "format_unsupported"
                            
                        elif 'chunk' in error_message or 'split' in error_message:
                            # Chunking error
                            error_id = log_chunking_error(
                                self.error_tracker, document_id, file_name, e, chunking_method,
                                file_path=file_path if not file_path.startswith('s3://') else None,
                                s3_key=file_path if file_path.startswith('s3://') else None
                            )
                            error_type = "chunking_error"
                            
                        else:
                            # Generic processing error
                            error_id = self.error_tracker.log_error(
                                document_id=document_id,
                                file_name=file_name,
                                error_type=ErrorType.PARSING_ERROR,
                                processing_stage=ProcessingStage.FILE_EXTRACTION,
                                error_message=f"Failed to process {file_name}: {str(e)}",
                                file_path=file_path if not file_path.startswith('s3://') else None,
                                s3_key=file_path if file_path.startswith('s3://') else None,
                                exception=e,
                                is_recoverable=True
                            )
                            error_type = "parsing_error"
                            
                        # Add detailed error to job status
                        job_status['errors'].append({
                            'file_path': file_path,
                            'file_name': file_name,
                            'error_type': error_type,
                            'error_id': str(error_id),
                            'error': str(e),
                            'is_recoverable': error_type != "password_protected"
                        })
                        
                    except Exception as tracking_error:
                        logger.error(f"Failed to track error for {file_path}: {tracking_error}")
                        # Fallback to basic error logging
                        job_status['errors'].append({
                            'file_path': file_path,
                            'error': str(e)
                        })
                    
                    # Still count as processed for progress calculation
                    job_status['files_processed'] += 1
            
            # Mark job as completed
            if job_status['status'] != 'cancelled':
                job_status['status'] = 'completed'
                logger.info(f"Job {job_id} completed successfully")
            
            job_status['end_time'] = time.time()
            job_status['duration'] = job_status['end_time'] - job_status['start_time']
            job_status['current_file'] = None
            
            # Log final summary
            logger.info(f"Job {job_id} finished - Status: {job_status['status']}, "
                       f"Processed: {job_status['files_processed']}, "
                       f"Failed: {job_status['files_failed']}, "
                       f"Duration: {job_status['duration']:.2f}s")
            
        except Exception as e:
            logger.error(f"Critical error in directory processing job {job_id}: {e}")
            job_status['status'] = 'failed'
            job_status['error'] = str(e)
            job_status['end_time'] = time.time()
            job_status['duration'] = job_status['end_time'] - job_status['start_time']
    
    def upload_and_process_s3(self, 
                             file, 
                             s3_path: str,
                             chunking_method: str,
                             chunk_size: int = 1000,
                             chunk_overlap: int = 200) -> Dict[str, Any]:
        """
        Upload file to S3 and process with specified chunking method
        """
        
        try:
            # Secure the filename
            filename = secure_filename(file.filename)
            if not filename:
                raise ValueError("Invalid filename")
            
            # Check file extension
            file_ext = Path(filename).suffix.lower()
            if file_ext not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Construct S3 key
            s3_key = f"{s3_path.rstrip('/')}/{filename}"
            
            # Upload to S3
            upload_result = self.s3_helper.upload_file_object(file, s3_key)
            if not upload_result:
                raise ValueError("Failed to upload file to S3")
            
            # Download content for processing
            content = self.s3_helper.get_file_content(s3_key)
            if content is None:
                raise ValueError("Failed to retrieve uploaded file content")
            
            # Get file info
            file_info = self.s3_helper.get_file_info(s3_key)
            file_size = file_info.get('ContentLength', 0) if file_info else 0
            content_type = file_info.get('ContentType', '') if file_info else ''
            
            # Process the file
            result = self._process_file_content(
                content=content,
                file_name=filename,
                file_path=s3_key,
                s3_key=s3_key,
                file_size=file_size,
                content_type=content_type,
                source_type='s3',
                chunking_method=chunking_method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            return {
                'status': 'success',
                'message': f'File uploaded and processed successfully',
                'document_id': result['document_id'],
                'chunks_created': result['chunks_created'],
                's3_key': s3_key,
                'file_name': filename,
                'chunking_method': chunking_method
            }
            
        except Exception as e:
            logger.error(f"Error uploading and processing S3 file: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _process_s3_file(self,
                        s3_key: str,
                        source_type: str,
                        chunking_method: str,
                        chunk_size: int,
                        chunk_overlap: int) -> Dict[str, Any]:
        """Process a single S3 file"""
        
        try:
            # Download content from S3
            content = self.s3_helper.get_file_content(s3_key)
            if content is None:
                raise ValueError(f"Failed to retrieve S3 file content: {s3_key}")
            
            # Get file info from S3
            file_info = self.s3_helper.get_file_info(s3_key)
            file_size = file_info.get('ContentLength', 0) if file_info else 0
            content_type = file_info.get('ContentType', '') if file_info else ''
            
            # Extract file name from S3 key
            file_name = os.path.basename(s3_key)
            
            return self._process_file_content(
                content=content,
                file_name=file_name,
                file_path=s3_key,  # Use S3 key as file path for S3 files
                s3_key=s3_key,
                file_size=file_size,
                content_type=content_type,
                source_type=source_type,
                chunking_method=chunking_method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error(f"Error processing S3 file {s3_key}: {e}")
            raise

    def _process_single_file(self, 
                           file_path: str,
                           source_type: str,
                           chunking_method: str,
                           chunk_size: int,
                           chunk_overlap: int,
                           s3_key: Optional[str] = None) -> Dict[str, Any]:
        """Process a single file"""
        
        # Read file content based on source type
        try:
            if source_type == 's3' or file_path.startswith('s3://'):
                # Handle S3 files
                if file_path.startswith('s3://'):
                    # Extract S3 key from s3:// URL
                    s3_key = file_path[5:]  # Remove 's3://' prefix
                elif s3_key:
                    # Use provided S3 key
                    pass
                else:
                    # Treat file_path as S3 key directly
                    s3_key = file_path
                
                logger.info(f"Reading S3 file content for key: {s3_key}")
                content = self.s3_helper.get_file_content(s3_key)
                
                # Get file info for S3
                file_name = os.path.basename(s3_key)
                file_size = len(content.encode('utf-8')) if isinstance(content, str) else len(content)
                content_type, _ = mimetypes.guess_type(file_name)
                
            else:
                # Handle local files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Get file info for local files
                file_name = os.path.basename(file_path)
                file_stat = os.stat(file_path)
                file_size = file_stat.st_size
                content_type, _ = mimetypes.guess_type(file_path)
                
        except Exception as e:
            raise ValueError(f"Could not read file {file_path}: {e}")
        
        return self._process_file_content(
            content=content,
            file_name=file_name,
            file_path=file_path,
            s3_key=s3_key,
            file_size=file_size,
            content_type=content_type,
            source_type=source_type,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def _process_file_content(self,
                             content: str,
                             file_name: str,
                             file_path: str,
                             source_type: str,
                             chunking_method: str,
                             chunk_size: int,
                             chunk_overlap: int,
                             s3_key: Optional[str] = None,
                             file_size: Optional[int] = None,
                             content_type: Optional[str] = None) -> Dict[str, Any]:
        """Process file content and store chunks"""
        
        # Determine file type
        file_ext = Path(file_name).suffix.lower()
        if file_ext == '.pdf':
            file_type = 'pdf'
        elif file_ext in ['.txt', '.md', '.markdown']:
            file_type = 'text'
        elif file_ext == '.json':
            file_type = 'json'
        elif file_ext in ['.csv', '.tsv']:
            file_type = 'csv'
        elif file_ext in ['.xlsx', '.xls']:
            file_type = 'excel'
        else:
            file_type = 'text'  # Default to text
        
        # Create document record
        document_id = self.db_helper.create_document_record(
            file_path=file_path,
            file_name=file_name,
            file_type=file_type,
            source_type=source_type,
            chunking_method=chunking_method,
            s3_key=s3_key,
            file_size=file_size,
            content_type=content_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        try:
            # Set chunking parameters
            self.document_processor.set_chunking_method(chunking_method)
            self.document_processor.chunk_size = chunk_size
            self.document_processor.chunk_overlap = chunk_overlap
            
            # Create metadata
            metadata = {
                'source_type': source_type,
                's3_key': s3_key,
                'file_path': file_path,
                'content_type': content_type,
                'file_type': file_type
            }
            
            # Process content based on file type
            if file_type == 'pdf':
                chunks = self.document_processor.process_pdf_content(content, s3_key or file_path)
            elif file_type == 'json':
                chunks = self.document_processor.process_json_content(content, s3_key or file_path)
            elif file_type == 'excel':
                # Excel files should be processed as text content, not adaptive chunking
                chunks = self.document_processor.process_text_content(content, s3_key or file_path, content_type)
            else:
                chunks = self.document_processor.process_text_content(content, s3_key or file_path, content_type)
            
            # Store chunks in database
            chunks_stored, failed_chunks = self.db_helper.store_document_chunks(document_id, chunks)
            
            # Log chunk-level failures if any
            if failed_chunks:
                logger.warning(f"DOCUMENT_CHUNK_FAILURES - {file_name}: {len(failed_chunks)} chunks failed to store out of {len(chunks)} total chunks")
                
                # Log detailed failure information
                for failed_chunk in failed_chunks:
                    logger.error(
                        f"CHUNK_FAILURE_DETAIL - Document: {file_name}, "
                        f"Chunk Index: {failed_chunk['chunk_index']}, "
                        f"Error Type: {failed_chunk['error_type']}, "
                        f"Error: {failed_chunk['error_message']}, "
                        f"Content Preview: {failed_chunk.get('content_preview', 'N/A')[:50]}..."
                    )
                
                # Group and summarize failures by type
                error_types = {}
                for failed_chunk in failed_chunks:
                    error_type = failed_chunk['error_type']
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                
                logger.warning(f"CHUNK_FAILURE_SUMMARY - {file_name}: {error_types}")
            
            # Update document status with chunk failure information
            if failed_chunks:
                error_summary = f"Stored {chunks_stored}/{len(chunks)} chunks. Failed chunks by type: {dict((error_type, count) for error_type, count in [(failed_chunk['error_type'], 1) for failed_chunk in failed_chunks])}"
                # Still mark as completed if most chunks succeeded, but include failure info
                if chunks_stored > 0:
                    self.db_helper.update_document_status(document_id, 'completed', chunks_stored)
                    logger.warning(f"Document {file_name} marked as completed with partial chunk failures: {error_summary}")
                else:
                    # If no chunks stored, mark as failed
                    self.db_helper.update_document_status(document_id, 'failed', error_message=f"All chunks failed to store: {error_summary}")
                    logger.error(f"Document {file_name} marked as failed - no chunks stored: {error_summary}")
            else:
                # All chunks stored successfully
                self.db_helper.update_document_status(document_id, 'completed', chunks_stored)
            
            logger.info(f"Successfully processed {file_name}: {chunks_stored}/{len(chunks)} chunks stored")
            
            return {
                'document_id': document_id,
                'chunks_created': chunks_stored,
                'chunks_failed': len(failed_chunks),
                'total_chunks_attempted': len(chunks),
                'failed_chunks_details': failed_chunks if failed_chunks else None,
                'file_name': file_name,
                'chunking_method': chunking_method
            }
            
        except Exception as e:
            # Update document status as failed
            self.db_helper.update_document_status(document_id, 'failed', error_message=str(e))
            raise e
    
    def get_all_job_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get all job statuses"""
        return self.processing_jobs.copy()
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a processing job (mark as cancelled, actual cancellation may take time)"""
        if job_id in self.processing_jobs:
            self.processing_jobs[job_id]['status'] = 'cancelled'
            return True
        return False
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old job statuses"""
        current_time = time.time()
        jobs_to_remove = []
        
        for job_id, job_status in self.processing_jobs.items():
            job_age = current_time - job_status['start_time']
            if job_age > (max_age_hours * 3600):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.processing_jobs[job_id]
        
        logger.info(f"Cleaned up {len(jobs_to_remove)} old job statuses")
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return list(self.supported_extensions)
    
    def validate_chunking_method(self, method: str) -> bool:
        """Validate if chunking method is supported"""
        supported_methods = ['recursive', 'semantic', 'agentic', 'adaptive', 'sentence']
        return method.lower() in supported_methods
    
    def get_available_chunking_methods(self) -> list:
        """Get list of available chunking methods with descriptions"""
        return [
            {
                'name': 'recursive',
                'label': 'Recursive Text Splitting',
                'description': 'Splits text recursively by different characters',
                'parameters': {
                    'chunk_size': {'default': 1000, 'min': 100, 'max': 4000},
                    'chunk_overlap': {'default': 200, 'min': 0, 'max': 500}
                }
            },
            {
                'name': 'semantic',
                'label': 'Semantic Chunking',
                'description': 'Groups text by semantic similarity',
                'parameters': {
                    'chunk_size': {'default': 1000, 'min': 100, 'max': 4000},
                    'chunk_overlap': {'default': 200, 'min': 0, 'max': 500}
                }
            },
            {
                'name': 'agentic',
                'label': 'Agentic Chunking',
                'description': 'AI-driven adaptive chunking strategy',
                'parameters': {
                    'chunk_size': {'default': 1000, 'min': 100, 'max': 4000},
                    'chunk_overlap': {'default': 200, 'min': 0, 'max': 500}
                }
            },
            {
                'name': 'adaptive',
                'label': 'Adaptive Chunking',
                'description': 'Dynamically adjusts chunk size based on content',
                'parameters': {
                    'chunk_size': {'default': 1000, 'min': 100, 'max': 4000},
                    'chunk_overlap': {'default': 200, 'min': 0, 'max': 500}
                }
            },
            {
                'name': 'sentence',
                'label': 'Sentence-based Chunking',
                'description': 'Splits text by sentence boundaries',
                'parameters': {
                    'chunk_size': {'default': 1000, 'min': 100, 'max': 4000},
                    'chunk_overlap': {'default': 200, 'min': 0, 'max': 500}
                }
            }
        ]
