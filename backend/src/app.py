from flask import Flask, jsonify, request
from flask_cors import CORS
from src.utils.s3_helper import S3Helper
from src.utils.llm_helper import LLMHelper
import os

app = Flask(__name__)
CORS(app)

s3_helper = S3Helper()

# Configure LLM Helper
llm_provider = os.environ.get('LLM_PROVIDER', 'gemini')
llm_helper = LLMHelper(provider=llm_provider)

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
            response = llm_helper.generate_response(prompt, include_s3_context, s3_query)
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
            result = llm_helper.query_s3_files(query, file_pattern, max_files)
            print("Query completed successfully", flush=True)
        except Exception as e:
            import traceback
            print(f"File query error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({'error': f'File query error: {str(e)}', 'traceback': traceback.format_exc()}), 500

        # Handle both old string response and new dict response formats
        if isinstance(result, dict):
            return jsonify({
                'response': result.get('response', ''), 
                'full_prompt': result.get('full_prompt', '')
            })
        else:
            # Fallback for old string response format
            return jsonify({'response': result, 'full_prompt': query})

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
