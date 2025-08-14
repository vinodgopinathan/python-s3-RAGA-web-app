import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import traceback

# Local imports
from utils.chunking_service import ChunkingService
from utils.enhanced_vector_db_helper import EnhancedVectorDBHelper

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configure file upload limits
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
chunking_service = ChunkingService()
db_helper = EnhancedVectorDBHelper()

@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def too_large(e):
    return jsonify({
        'status': 'error',
        'message': 'File too large. Maximum size is 100MB.'
    }), 413

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    logger.error(traceback.format_exc())
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

# ============ DIRECTORY PROCESSING ENDPOINTS ============

@app.route('/api/process-directory', methods=['POST'])
def process_directory():
    """Process all files in a directory with specified chunking method"""
    try:
        data = request.get_json()
        
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
            recursive=recursive
        )
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in process_directory: {e}")
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
        logger.error(f"Error getting job status: {e}")
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
        logger.error(f"Error getting all jobs: {e}")
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
        logger.error(f"Error cancelling job: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to cancel job'
        }), 500

# ============ S3 UPLOAD ENDPOINTS ============

@app.route('/api/upload-s3', methods=['POST'])
def upload_to_s3():
    """Upload file to S3 and process with specified chunking method"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file provided'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Get form data
        s3_path = request.form.get('s3_path', '').strip()
        chunking_method = request.form.get('chunking_method', '').lower()
        
        if not s3_path:
            return jsonify({
                'status': 'error',
                'message': 'S3 path is required'
            }), 400
        
        if not chunking_method:
            return jsonify({
                'status': 'error',
                'message': 'Chunking method is required'
            }), 400
        
        # Validate chunking method
        if not chunking_service.validate_chunking_method(chunking_method):
            return jsonify({
                'status': 'error',
                'message': f'Invalid chunking method: {chunking_method}'
            }), 400
        
        # Optional parameters
        chunk_size = int(request.form.get('chunk_size', 1000))
        chunk_overlap = int(request.form.get('chunk_overlap', 200))
        
        # Validate parameters
        if chunk_size < 100:
            return jsonify({
                'status': 'error',
                'message': 'chunk_size must be >= 100'
            }), 400
        
        if chunk_overlap < 0:
            return jsonify({
                'status': 'error',
                'message': 'chunk_overlap must be >= 0'
            }), 400
        
        # Process upload
        result = chunking_service.upload_and_process_s3(
            file=file,
            s3_path=s3_path,
            chunking_method=chunking_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Error in upload_to_s3: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to upload and process file'
        }), 500

# ============ DATABASE QUERY ENDPOINTS ============

@app.route('/api/search', methods=['POST'])
def search_documents():
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
        source_type = data.get('source_type')
        min_similarity = data.get('min_similarity', 0.0)
        
        # Perform search
        results = db_helper.similarity_search(
            query=query,
            k=k,
            chunking_method=chunking_method,
            source_type=source_type,
            min_similarity=min_similarity
        )
        
        return jsonify({
            'status': 'success',
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Search failed'
        }), 500

@app.route('/api/search-text', methods=['POST'])
def search_text():
    """Full-text search on document content"""
    try:
        data = request.get_json()
        search_term = data.get('search_term', '').strip()
        
        if not search_term:
            return jsonify({
                'status': 'error',
                'message': 'Search term is required'
            }), 400
        
        limit = data.get('limit', 10)
        chunking_method = data.get('chunking_method')
        
        results = db_helper.search_by_content(
            search_term=search_term,
            limit=limit,
            chunking_method=chunking_method
        )
        
        return jsonify({
            'status': 'success',
            'search_term': search_term,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Text search failed'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        stats = db_helper.get_document_stats()
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get statistics'
        }), 500

@app.route('/api/recent-documents', methods=['GET'])
def get_recent_documents():
    """Get recently processed documents"""
    try:
        limit = request.args.get('limit', 20, type=int)
        documents = db_helper.get_recent_documents(limit)
        
        return jsonify({
            'status': 'success',
            'documents': documents,
            'count': len(documents)
        })
    except Exception as e:
        logger.error(f"Error getting recent documents: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to get recent documents'
        }), 500

@app.route('/api/delete-document/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    """Delete a document and its chunks"""
    try:
        success = db_helper.delete_document(document_id)
        if success:
            return jsonify({
                'status': 'success',
                'message': f'Document {document_id} deleted'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Document not found'
            }), 404
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to delete document'
        }), 500

# ============ UTILITY ENDPOINTS ============

@app.route('/api/chunking-methods', methods=['GET'])
def get_chunking_methods():
    """Get available chunking methods"""
    return jsonify({
        'status': 'success',
        'methods': [
            {
                'value': 'recursive',
                'label': 'Recursive (Structure-based)',
                'description': 'Uses hierarchical separators for structured content'
            },
            {
                'value': 'semantic',
                'label': 'Semantic (Meaning-based)',
                'description': 'Uses sentence embeddings to find natural boundaries'
            },
            {
                'value': 'agentic',
                'label': 'Agentic (LLM-guided)',
                'description': 'Uses LLM intelligence for optimal boundaries'
            },
            {
                'value': 'adaptive',
                'label': 'Adaptive (Auto-selection)',
                'description': 'Automatically chooses the best method'
            },
            {
                'value': 'sentence',
                'label': 'Sentence-based (Legacy)',
                'description': 'Simple sentence-based chunking'
            }
        ]
    })

@app.route('/api/supported-extensions', methods=['GET'])
def get_supported_extensions():
    """Get supported file extensions"""
    return jsonify({
        'status': 'success',
        'extensions': chunking_service.get_supported_extensions()
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Chunking service is running',
        'supported_methods': ['recursive', 'semantic', 'agentic', 'adaptive', 'sentence']
    })

if __name__ == '__main__':
    # Clean up old jobs on startup
    chunking_service.cleanup_old_jobs()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    )
