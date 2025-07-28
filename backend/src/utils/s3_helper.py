import boto3
import os
from werkzeug.utils import secure_filename
import io
import json
import PyPDF2

class S3Helper:
    def __init__(self):
        print("Initializing S3Helper with AWS credentials...")
        try:
            self.s3 = boto3.client(
                's3',
                aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
            )
            self.bucket_name = os.environ.get('S3_BUCKET_NAME')
            print(f"Successfully initialized S3 client. Bucket name: {self.bucket_name}")
            # Test connection
            self.s3.head_bucket(Bucket=self.bucket_name)
            print("Successfully connected to S3 bucket")
        except Exception as e:
            print(f"Error initializing S3 client: {str(e)}")
            raise

    def list_files(self):
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name)
            files = []
            if 'Contents' in response:
                for item in response['Contents']:
                    files.append({
                        'key': item['Key'],
                        'size': item['Size'],
                        'last_modified': item['LastModified'].isoformat()
                    })
            return files
        except Exception as e:
            print(f"Error listing files: {str(e)}")
            raise

    def upload_file(self, file):
        try:
            print(f"Starting upload for file: {file.filename}")
            filename = secure_filename(file.filename)
            print(f"Secured filename: {filename}")
            
            # Test bucket access before uploading
            try:
                self.s3.head_bucket(Bucket=self.bucket_name)
                print(f"Successfully verified bucket access for: {self.bucket_name}")
            except Exception as e:
                print(f"Error accessing bucket: {str(e)}")
                return False
            
            # Attempt upload
            print(f"Uploading to bucket: {self.bucket_name}")
            self.s3.upload_fileobj(
                file,
                self.bucket_name,
                filename,
                ExtraArgs={'ACL': 'private'}
            )
            print(f"Upload completed successfully for: {filename}")
            return True
        except Exception as e:
            print(f"Error uploading file: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False

    def get_file_content(self, file_key):
        """Get the content of a file from S3"""
        try:
            print(f"Getting content for file: {file_key}")
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)
            
            # Handle different file types
            if file_key.endswith('.pdf'):
                return self._extract_pdf_text(response['Body'])
            else:
                # For text files, decode the content
                content = response['Body'].read().decode('utf-8')
                print(f"Successfully retrieved content for {file_key} (length: {len(content)})")
                return content
        except Exception as e:
            print(f"Error getting file content for {file_key}: {str(e)}")
            raise

    def _extract_pdf_text(self, pdf_stream):
        """Extract text content from PDF file"""
        try:
            import PyPDF2
            
            # Read the stream into memory for PyPDF2
            pdf_content = pdf_stream.read()
            pdf_file = io.BytesIO(pdf_content)
            
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text_content = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n"
            
            print(f"Successfully extracted text from PDF (length: {len(text_content)})")
            return text_content
        except Exception as e:
            print(f"Error extracting PDF text: {str(e)}")
            # Return empty string if PDF extraction fails
            return ""

    def search_files(self, search_query, file_extensions=None):
        """Search for files based on query and optionally filter by file extensions"""
        try:
            files = self.list_files()
            matching_files = []
            
            for file_info in files:
                file_key = file_info['key']
                
                # Filter by file extension if specified
                if file_extensions:
                    if not any(file_key.endswith(ext) for ext in file_extensions):
                        continue
                
                # Check if search query matches filename
                if search_query.lower() in file_key.lower():
                    matching_files.append(file_info)
                    continue
                
                # For text files and PDFs, search content
                try:
                    if file_key.endswith(('.txt', '.md', '.json', '.csv', '.log', '.pdf')):
                        content = self.get_file_content(file_key)
                        if search_query.lower() in content.lower():
                            file_info['match_type'] = 'content'
                            matching_files.append(file_info)
                except Exception as e:
                    print(f"Error searching content of {file_key}: {str(e)}")
                    continue
            
            return matching_files
        except Exception as e:
            print(f"Error searching files: {str(e)}")
            raise

    def get_file_summary(self, file_key):
        """Get a summary of file information"""
        try:
            # Get file metadata
            response = self.s3.head_object(Bucket=self.bucket_name, Key=file_key)
            
            summary = {
                'key': file_key,
                'size': response['ContentLength'],
                'last_modified': response['LastModified'].isoformat(),
                'content_type': response.get('ContentType', 'unknown'),
                'metadata': response.get('Metadata', {})
            }
            
            # Add content preview for text files and PDFs
            if file_key.endswith(('.txt', '.md', '.json', '.csv', '.log', '.pdf')):
                try:
                    content = self.get_file_content(file_key)
                    summary['content_preview'] = content[:500] + ('...' if len(content) > 500 else '')
                    summary['content_length'] = len(content)
                except Exception as e:
                    summary['content_preview'] = f"Error reading content: {str(e)}"
            
            return summary
        except Exception as e:
            print(f"Error getting file summary: {str(e)}")
            raise
