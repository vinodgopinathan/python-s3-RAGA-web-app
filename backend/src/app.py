from flask import Flask, jsonify, request
from flask_cors import CORS
from utils.s3_helper import S3Helper
from utils.rag_llm_helper import RAGLLMHelper
from utils.chunking_service import ChunkingService
from utils.vector_db_helper import VectorDBHelper
from utils.raga_helper import RAGAHelper
import os
import logging
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

# Set up logger
logger = logging.getLogger(__name__)
import traceback

# Force logging to stdout for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# Configure file upload limits
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

s3_helper = S3Helper()

# Initialize services
chunking_service = ChunkingService()
db_helper = VectorDBHelper()

# Configure RAG LLM Helper
llm_provider = os.environ.get('LLM_PROVIDER', 'gemini')
print(f"DEBUG: app.py - LLM_PROVIDER from env: {llm_provider}")
print(f"DEBUG: app.py - About to create RAGLLMHelper with provider: {llm_provider}")
rag_llm_helper = RAGLLMHelper(provider=llm_provider)
print(f"DEBUG: app.py - RAGLLMHelper created successfully")

# Configure RAGA Helper
raga_helper = RAGAHelper(db_helper, s3_helper)
print(f"DEBUG: app.py - RAGAHelper created successfully")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

@app.route('/api/health', methods=['GET'])
def api_health_check():
    return jsonify({'status': 'ok'}), 200

@app.route('/api/files', methods=['GET'])
def list_files():
    try:
        print("API call received: GET /api/files", flush=True)
        files = s3_helper.list_files()
        print(f"Retrieved {len(files)} files", flush=True)
        return jsonify({'files': files})
    except Exception as e:
        print(f"Error in list_files: {str(e)}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            print("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return jsonify({'error': 'No selected file'}), 400

        print(f"Attempting to upload file: {file.filename}")
        success = s3_helper.upload_file(file)
        if success:
            print(f"Successfully uploaded file: {file.filename}")
            return jsonify({'message': 'File uploaded successfully'})
        else:
            print(f"Failed to upload file: {file.filename}")
            return jsonify({'error': 'Failed to upload file'}), 500
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/s3/list', methods=['GET'])
def list_s3_objects():
    """List objects in S3 bucket with optional prefix"""
    try:
        prefix = request.args.get('prefix', '')
        logger.info(f"Listing S3 objects with prefix: {prefix}")
        
        objects = s3_helper.list_objects(prefix=prefix if prefix else None)
        
        return jsonify({
            'status': 'success',
            'objects': objects,
            'count': len(objects),
            'prefix': prefix
        })
        
    except Exception as e:
        logger.error(f"Error listing S3 objects: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/upload-s3', methods=['POST'])
def upload_and_process_s3():
    """Upload file to S3 and process with chunking - with proper progress tracking"""
    try:
        # Check for file
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400

        # Get form parameters
        s3_path = request.form.get('s3_path', '').strip()
        chunking_method = request.form.get('chunking_method', 'recursive')
        chunk_size = int(request.form.get('chunk_size', 1000))
        chunk_overlap = int(request.form.get('chunk_overlap', 200))

        if not s3_path:
            return jsonify({'status': 'error', 'message': 'S3 path is required'}), 400

        print(f"Processing upload for file: {file.filename} to S3 path: {s3_path}")

        # Step 1: Upload file to S3
        try:
            # Create a temporary file-like object to upload
            from werkzeug.datastructures import FileStorage
            from io import BytesIO
            
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            # Create S3 key from s3_path and filename
            s3_key = f"{s3_path.rstrip('/')}/{file.filename}"
            
            # Upload to S3 directly using s3_helper
            upload_success = s3_helper.upload_file_content(file_content, s3_key)
            
            if not upload_success:
                return jsonify({
                    'status': 'error', 
                    'message': 'Failed to upload file to S3'
                }), 500
                
            print(f"Successfully uploaded {file.filename} to S3 key: {s3_key}")
            
        except Exception as upload_error:
            print(f"Error uploading to S3: {upload_error}")
            return jsonify({
                'status': 'error', 
                'message': f'S3 upload failed: {str(upload_error)}'
            }), 500

        # Step 2: Process the uploaded file
        try:
            result = chunking_service._process_single_file(
                file_path=f"s3://{s3_key}",
                source_type='s3',
                chunking_method=chunking_method,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            print(f"Successfully processed {file.filename}: {result['chunks_created']} chunks created")
            
            return jsonify({
                'status': 'success',
                'message': 'File uploaded and processed successfully',
                'document_id': result['document_id'],
                'chunks_created': result['chunks_created'],
                's3_key': s3_key,
                'file_name': file.filename,
                'chunking_method': chunking_method,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            })
            
        except Exception as processing_error:
            print(f"Error processing file: {processing_error}")
            # File was uploaded but processing failed
            return jsonify({
                'status': 'error',
                'message': f'File uploaded to S3 but processing failed: {str(processing_error)}',
                's3_key': s3_key,
                'file_name': file.filename
            }), 500

    except Exception as e:
        print(f"Error in upload_and_process_s3: {str(e)}")
        return jsonify({
            'status': 'error', 
            'message': f'Upload and processing failed: {str(e)}'
        }), 500

@app.route('/api/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        include_s3_context = data.get('include_s3_context', False)
        s3_query = data.get('s3_query')

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        try:
            response = rag_llm_helper.generate_response(prompt, include_s3_context, s3_query)
        except Exception as e:
            import traceback
            print(f"LLM error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'LLM error: {str(e)}', 'traceback': traceback.format_exc()}), 500

        return jsonify({'response': response})

    except Exception as e:
        import traceback
        print(f"API error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/query-files', methods=['POST'])
def query_files():
    """Query S3 files and generate LLM response based on their content"""
    try:
        print("API call received: POST /api/query-files", flush=True)
        data = request.get_json()
        query = data.get('query')
        file_pattern = data.get('file_pattern')  # Optional: comma-separated file extensions
        max_files = data.get('max_files', 5)

        print(f"Query: {query}, Pattern: {file_pattern}, Max files: {max_files}", flush=True)

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        try:
            result = rag_llm_helper.query_s3_files(query, file_pattern, max_files)
            print("Query completed successfully", flush=True)
        except Exception as e:
            import traceback
            print(f"File query error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'File query error: {str(e)}', 'traceback': traceback.format_exc()}), 500

        # Handle both old string response and new dict response formats
        if isinstance(result, dict):
            # Return enhanced hybrid search information
            response_data = {
                'response': result.get('response', ''), 
                'full_prompt': result.get('full_prompt', ''),
                'sources': result.get('sources', []),
                'context_used': result.get('context_used', False),
                'search_strategy': result.get('search_strategy', 'hybrid')
            }
            
            # Include query analysis and search stats if available
            if 'query_analysis' in result:
                response_data['query_analysis'] = result['query_analysis']
            
            if 'search_stats' in result:
                response_data['search_stats'] = result['search_stats']
                
            return jsonify(response_data)
        else:
            # Fallback for old string response format
            return jsonify({
                'response': result, 
                'full_prompt': query,
                'search_strategy': 'legacy'
            })

    except Exception as e:
        import traceback
        print(f"API error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/search-files', methods=['POST'])
def search_files():
    """Search for files in S3 bucket"""
    try:
        data = request.get_json()
        search_query = data.get('query')
        file_extensions = data.get('file_extensions')  # Optional: list of extensions

        if not search_query:
            return jsonify({'error': 'Search query is required'}), 400

        try:
            matching_files = s3_helper.search_files(search_query, file_extensions)
            return jsonify({'files': matching_files})
        except Exception as e:
            import traceback
            print(f"Search error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'Search error: {str(e)}', 'traceback': traceback.format_exc()}), 500

    except Exception as e:
        import traceback
        print(f"API error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/file-content/<path:file_key>', methods=['GET'])
def get_file_content(file_key):
    """Get content of a specific file from S3"""
    try:
        try:
            content = s3_helper.get_file_content(file_key)
            return jsonify({'content': content, 'file_key': file_key})
        except Exception as e:
            import traceback
            print(f"File content error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'File content error: {str(e)}', 'traceback': traceback.format_exc()}), 500

    except Exception as e:
        import traceback
        print(f"API error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/file-summary/<path:file_key>', methods=['GET'])
def get_file_summary(file_key):
    """Get summary of a specific file from S3"""
    try:
        try:
            summary = s3_helper.get_file_summary(file_key)
            return jsonify({'summary': summary})
        except Exception as e:
            import traceback
            print(f"File summary error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'File summary error: {str(e)}', 'traceback': traceback.format_exc()}), 500

    except Exception as e:
        import traceback
        print(f"API error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# New RAG-specific endpoints

@app.route('/api/rag/index-document', methods=['POST'])
def index_document():
    """Index a document for RAG"""
    try:
        data = request.get_json()
        s3_key = data.get('s3_key')
        force_reindex = data.get('force_reindex', False)

        if not s3_key:
            return jsonify({'error': 'S3 key is required'}), 400

        # Use synchronous method directly (no async needed anymore)
        result = rag_llm_helper.index_document(s3_key, force_reindex)
        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"Index document error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/rag/search', methods=['POST'])
def rag_search():
    """Search documents using RAG vector similarity"""
    try:
        data = request.get_json()
        query = data.get('query')
        limit = data.get('limit', 5)
        similarity_threshold = data.get('similarity_threshold', 0.7)

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Use synchronous method directly (no async needed anymore)
        result = rag_llm_helper.search_documents(query, limit, similarity_threshold)
        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"RAG search error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/rag/query', methods=['POST'])
def rag_query():
    """Query files using RAG (Retrieval-Augmented Generation)"""
    try:
        data = request.get_json()
        query = data.get('query')
        auto_index = data.get('auto_index', True)
        similarity_threshold = data.get('similarity_threshold', 0.7)
        max_chunks = data.get('max_chunks', 5)

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Run async function in event loop
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                rag_llm_helper.query_s3_files_rag(
                    query=query,
                    auto_index=auto_index,
                    similarity_threshold=similarity_threshold,
                    max_chunks=max_chunks
                )
            )
            return jsonify(result)
        finally:
            loop.close()

    except Exception as e:
        import traceback
        print(f"RAG query error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/rag/index-all', methods=['POST'])
def index_all_documents():
    """Index all documents from S3 bucket"""
    try:
        # Use RAGLLMHelper to reindex all documents
        result = rag_llm_helper.reindex_all_documents()
        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"Index all error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """
    Search documents using hybrid search - Frontend compatibility endpoint
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query']
        k = data.get('k', 10)  # Number of results to return
        
        logger.info(f"Frontend search request for query: {query}")
        
        # Import here to avoid circular imports
        from utils.hybrid_search_helper import HybridSearchHelper
        
        # Initialize hybrid search helper
        hybrid_helper = HybridSearchHelper(db_helper)
        
        # Perform hybrid search
        import asyncio
        result = asyncio.run(hybrid_helper.hybrid_search_and_answer(query, max_results=k))
        
        # Transform the result to match frontend expectations
        frontend_result = {
            'results': [],
            'total': len(result.get('sources', [])),
            'query': query,
            'response': result.get('response', ''),
            'search_strategy': result.get('search_strategy', 'hybrid')
        }
        
        # Transform sources to match frontend format
        for source in result.get('sources', []):
            frontend_result['results'].append({
                'content': source.get('content_preview', ''),
                'source': source.get('s3_key', ''),
                'chunk_index': source.get('chunk_index', 0),
                'score': source.get('combined_score', 0),
                'search_type': source.get('search_type', 'hybrid')
            })
        
        logger.info(f"Frontend search completed with {len(frontend_result['results'])} results")
        
        return jsonify(frontend_result)
        
    except Exception as e:
        logger.error(f"Error in frontend search: {str(e)}")
        return jsonify({
            'results': [],
            'total': 0,
            'error': str(e),
            'response': f"I encountered an error while searching: {str(e)}"
        }), 500

@app.route('/api/rag/hybrid-search', methods=['POST'])
def hybrid_search():
    """Query files using advanced hybrid search (keyword + semantic)"""
    try:
        data = request.get_json()
        query = data.get('query')
        max_results = data.get('max_results', 10)
        semantic_threshold = data.get('semantic_threshold', 0.6)
        include_stats = data.get('include_stats', True)
        include_analysis = data.get('include_analysis', True)

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        # Run async hybrid search
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            logger.info(f"Starting hybrid search for query: {query}")
            result = loop.run_until_complete(
                rag_llm_helper.query_s3_files_hybrid(
                    query=query,
                    max_results=max_results,
                    semantic_threshold=semantic_threshold
                )
            )
            logger.info(f"Hybrid search completed, result keys: {result.keys()}")
            
            # Filter response based on client preferences
            filtered_result = {
                'response': result.get('response', ''),
                'sources': result.get('sources', []),
                'context_used': result.get('context_used', False),
                'search_strategy': result.get('search_strategy', 'hybrid')
            }
            
            if include_analysis and 'query_analysis' in result:
                filtered_result['query_analysis'] = result['query_analysis']
            
            if include_stats and 'search_stats' in result:
                filtered_result['search_stats'] = result['search_stats']
            
            logger.info(f"Returning filtered result with keys: {filtered_result.keys()}")
            return jsonify(filtered_result)
            
        finally:
            loop.close()

    except Exception as e:
        import traceback
        print(f"Hybrid search error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/rag/stats', methods=['GET'])
def get_rag_stats():
    """Get RAG database statistics"""
    try:
        # Get statistics from the vector database
        database_stats = db_helper.get_database_stats()
        document_stats = db_helper.get_document_stats()
        
        # Combine stats in expected format
        stats = {
            'database': database_stats,
            'documents': document_stats
        }
        
        return jsonify(stats)

    except Exception as e:
        import traceback
        print(f"Get stats error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get stats for frontend - wrapper for RAG stats"""
    try:
        # Get statistics from the vector database
        document_stats = db_helper.get_document_stats()
        
        # Extract overall stats from the nested structure
        overall_stats = document_stats.get('overall', {})
        
        # Format for frontend expectations
        stats = {
            'overall': {
                'total_documents': overall_stats.get('total_documents', 0),
                'total_chunks': overall_stats.get('total_chunks', 0),
                'avg_chunk_length': overall_stats.get('avg_chunk_length', 0),
                'completed_docs': overall_stats.get('completed_docs', 0),
                'failed_docs': overall_stats.get('failed_docs', 0),
                'processing_docs': overall_stats.get('processing_docs', 0)
            },
            'by_chunking_method': document_stats.get('by_chunking_method', []),
            'by_source_type': document_stats.get('by_source_type', [])
        }
        
        # Wrap the response to match frontend expectations
        return jsonify({'stats': stats})

    except Exception as e:
        import traceback
        print(f"Get stats error: {str(e)}")
        print(traceback.format_exc())
        # Return default empty stats if there's an error
        return jsonify({
            'stats': {
                'overall': {
                    'total_documents': 0,
                    'total_chunks': 0,
                    'avg_chunk_length': 0,
                    'completed_docs': 0,
                    'failed_docs': 0,
                    'processing_docs': 0
                },
                'by_chunking_method': [],
                'by_source_type': []
            }
        })

@app.route('/api/rag/documents', methods=['GET'])
def list_indexed_documents():
    """List all indexed documents"""
    try:
        # Use RAGLLMHelper to list documents
        result = rag_llm_helper.list_documents()
        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"List documents error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

# ============ ERROR HANDLERS ============

@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum size is 100MB.'
    }), 413

# ============ CHUNKING ENDPOINTS ============

@app.route('/api/process-directory', methods=['POST'])
def process_directory():
    """Process all files in a directory with specified chunking method"""
    try:
        data = request.get_json()
        print(f"DEBUG: Received request data: {data}", flush=True)
        
        # Validate required fields
        required_fields = ['directory_path', 'chunking_method']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        directory_path = data['directory_path']
        chunking_method = data['chunking_method'].lower()
        
        # Validate chunking method
        if not chunking_service.validate_chunking_method(chunking_method):
            return jsonify({
                'status': 'error',
                'message': f'Invalid chunking method: {chunking_method}'
            }), 400
        
        # Optional parameters
        chunk_size = data.get('chunk_size', 1000)
        chunk_overlap = data.get('chunk_overlap', 200)
        recursive = data.get('recursive', True)
        # Always use S3 - no source_type parameter needed
        
        print(f"DEBUG: Processing with source_type=s3, chunking_method={chunking_method}", flush=True)
        
        # Validate parameters
        if not isinstance(chunk_size, int) or chunk_size < 100:
            return jsonify({
                'status': 'error',
                'message': 'chunk_size must be an integer >= 100'
            }), 400
        
        if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
            return jsonify({
                'status': 'error',
                'message': 'chunk_overlap must be a non-negative integer'
            }), 400
        
        # Start directory processing
        result = chunking_service.process_directory(
            directory_path=directory_path,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            recursive=recursive,
            source_type='s3'  # Always use S3
        )
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        print(f"Error in process_directory: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to start directory processing'
        }), 500

@app.route('/api/job-status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of a processing job"""
    try:
        status = chunking_service.get_job_status(job_id)
        if status is None:
            return jsonify({
                'status': 'error',
                'message': 'Job not found'
            }), 404
        
        return jsonify({
            'status': 'success',
            'job_status': status
        })
        
    except Exception as e:
        print(f"Error getting job status: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get job status'
        }), 500

@app.route('/api/all-jobs', methods=['GET'])
def get_all_jobs():
    """Get all job statuses"""
    try:
        jobs = chunking_service.get_all_job_statuses()
        return jsonify({
            'status': 'success',
            'jobs': jobs
        })
    except Exception as e:
        print(f"Error getting all jobs: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get job statuses'
        }), 500

@app.route('/api/cancel-job/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a processing job"""
    try:
        success = chunking_service.cancel_job(job_id)
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Job {job_id} cancelled'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Job not found'
            }), 404
    except Exception as e:
        print(f"Error cancelling job: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to cancel job'
        }), 500

@app.route('/api/chunking-methods', methods=['GET'])
def get_chunking_methods():
    """Get available chunking methods and their parameters"""
    try:
        methods = chunking_service.get_available_chunking_methods()
        return jsonify({
            'status': 'success',
            'methods': methods
        })
    except Exception as e:
        logger.error(f"Error getting chunking methods: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/errors', methods=['GET'])
def get_processing_errors():
    """Get processing errors for debugging"""
    try:
        # Get error summary
        summary = chunking_service.error_tracker.get_error_summary()
        return jsonify({
            'status': 'success',
            'summary': summary
        })
    except Exception as e:
        logger.error(f"Error getting processing errors: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/test-error-tracking', methods=['POST'])
def test_error_tracking():
    """Test error tracking functionality"""
    try:
        from utils.error_tracker import ErrorType, ProcessingStage
        import uuid
        
        # Test error tracking with known problematic file
        document_id = uuid.uuid4()
        error_id = chunking_service.error_tracker.log_error(
            document_id=document_id,
            file_name="test_file.xlsx",
            error_type=ErrorType.PARSING_ERROR,
            processing_stage=ProcessingStage.FILE_EXTRACTION,
            error_message="Test error for validation",
            s3_key="test_file.xlsx",
            is_recoverable=True
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Error tracking test completed',
            'error_id': str(error_id),
            'document_id': str(document_id)
        })
    except Exception as e:
        logger.error(f"Error testing error tracking: {e}")
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/supported-extensions', methods=['GET'])
def get_supported_extensions():
    """Get supported file extensions"""
    return jsonify({
        'status': 'success',
        'extensions': chunking_service.get_supported_extensions()
    })

@app.route('/api/supported-file-types', methods=['GET'])
def get_supported_file_types():
    """Get supported file types with library availability"""
    try:
        # Get supported types from S3Helper
        supported_types = s3_helper.get_supported_file_types()
        
        return jsonify({
            'status': 'success',
            'supported_types': supported_types,
            'total_extensions': sum(len(exts) for exts in supported_types.values())
        })
    except Exception as e:
        logger.error(f"Error getting supported file types: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/vector-search', methods=['POST'])
def vector_search():
    """Search documents using vector similarity"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Query is required'
            }), 400
        
        # Optional parameters
        k = data.get('k', 5)
        chunking_method = data.get('chunking_method')
        min_similarity = data.get('min_similarity', 0.0)
        
        # Perform search (always use S3)
        results = db_helper.similarity_search(
            query=query,
            k=k,
            chunking_method=chunking_method,
            source_type='s3',  # Always use S3
            min_similarity=min_similarity
        )
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        print(f"Error in vector search: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Vector search failed'
        }), 500

@app.route('/api/chunking-stats', methods=['GET'])
def get_chunking_stats():
    """Get database statistics"""
    try:
        stats = db_helper.get_document_stats()
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    except Exception as e:
        print(f"Error getting stats: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get statistics'
        }), 500

@app.route('/api/errors/summary', methods=['GET'])
def get_error_summary():
    """Get summary of all processing errors"""
    try:
        from utils.error_tracker import ErrorTracker
        error_tracker = ErrorTracker(db_helper)
        summary = error_tracker.get_error_summary()
        
        return jsonify({
            'status': 'success',
            'error_summary': summary
        })
    except Exception as e:
        print(f"Error getting error summary: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get error summary'
        }), 500

@app.route('/api/errors/password-protected', methods=['GET'])
def get_password_protected_files():
    """Get all password-protected file errors"""
    try:
        from utils.error_tracker import ErrorTracker
        error_tracker = ErrorTracker(db_helper)
        password_files = error_tracker.get_password_protected_files()
        
        return jsonify({
            'status': 'success',
            'password_protected_files': password_files,
            'count': len(password_files)
        })
    except Exception as e:
        print(f"Error getting password-protected files: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get password-protected files'
        }), 500

@app.route('/api/errors/recoverable', methods=['GET'])
def get_recoverable_errors():
    """Get errors that can be retried"""
    try:
        from utils.error_tracker import ErrorTracker
        error_tracker = ErrorTracker(db_helper)
        max_retries = request.args.get('max_retries', 3, type=int)
        recoverable_errors = error_tracker.get_recoverable_errors(max_retries)
        
        return jsonify({
            'status': 'success',
            'recoverable_errors': recoverable_errors,
            'count': len(recoverable_errors)
        })
    except Exception as e:
        print(f"Error getting recoverable errors: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get recoverable errors'
        }), 500

@app.route('/api/errors/document/<document_id>', methods=['GET'])
def get_document_errors(document_id):
    """Get all errors for a specific document"""
    try:
        from utils.error_tracker import ErrorTracker
        import uuid
        
        error_tracker = ErrorTracker(db_helper)
        document_uuid = uuid.UUID(document_id)
        errors = error_tracker.get_errors_by_document(document_uuid)
        
        return jsonify({
            'status': 'success',
            'document_id': document_id,
            'errors': errors,
            'count': len(errors)
        })
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Invalid document ID format'
        }), 400
    except Exception as e:
        print(f"Error getting document errors: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get document errors'
        }), 500

@app.route('/api/errors/<error_id>/resolve', methods=['POST'])
def resolve_error(error_id):
    """Mark an error as resolved"""
    try:
        from utils.error_tracker import ErrorTracker
        import uuid
        
        data = request.get_json()
        resolution_notes = data.get('resolution_notes', 'Manually resolved')
        resolved_by = data.get('resolved_by', 'user')
        
        error_tracker = ErrorTracker(db_helper)
        error_uuid = uuid.UUID(error_id)
        success = error_tracker.resolve_error(error_uuid, resolution_notes, resolved_by)
        
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Error marked as resolved',
                'error_id': error_id
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Error not found or could not be resolved'
            }), 404
            
    except ValueError:
        return jsonify({
            'status': 'error',
            'message': 'Invalid error ID format'
        }), 400
    except Exception as e:
        print(f"Error resolving error: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to resolve error'
        }), 500

@app.route('/api/raga/stats', methods=['GET'])
def get_raga_system_stats():
    """Get comprehensive RAGA system statistics"""
    try:
        print("Getting RAGA system stats...")
        
        import asyncio
        stats = asyncio.run(raga_helper.get_raga_stats())
        
        print("RAGA stats retrieved successfully")
        return jsonify(stats)
        
    except Exception as e:
        import traceback
        print(f"RAGA stats error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    import sys
    import logging
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Force stdout to be unbuffered
    sys.stdout.reconfigure(line_buffering=True)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
