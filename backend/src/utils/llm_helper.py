import os
import google.generativeai as genai
from .s3_helper import S3Helper

# Conditional OpenAI import to avoid initialization issues when using Gemini
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class LLMHelper:
    def __init__(self, provider=None):
        self.provider = provider or os.environ.get('PROVIDER', 'gemini')
        self.s3_helper = S3Helper()
        
        if self.provider == 'gemini':
            self.api_key = os.environ.get('GEMINI_API_KEY')
            self.model_name = os.environ.get('MODEL')
            print(f"Initializing LLMHelper with provider: {self.provider}, model: {self.model_name}")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        elif self.provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI library not available. Please install openai package.")
            self.api_key = os.environ.get('OPENAI_API_KEY')
            self.model_name = os.environ.get('MODEL')
            print(f"Initializing LLMHelper with provider: {self.provider}, model: {self.model_name}")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_response(self, prompt, include_s3_context=False, s3_query=None):
        """Generate response with optional S3 context"""
        if include_s3_context and s3_query:
            # Enhance prompt with S3 file context
            prompt = self._enhance_prompt_with_s3_context(prompt, s3_query)
        
        if self.provider == 'gemini':
            return self._generate_gemini_response(prompt)
        elif self.provider == 'openai':
            return self._generate_openai_response(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def query_s3_files(self, query, file_pattern=None, max_files=5):
        """Query S3 files and generate response based on their content"""
        try:
            print(f"Querying S3 files with query: {query}")
            
            # Extract search terms from natural language queries
            search_terms = self._extract_search_terms(query)
            print(f"Extracted search terms: {search_terms}")
            
            # Search for relevant files using extracted terms
            all_matching_files = []
            for term in search_terms:
                if file_pattern:
                    extensions = [ext.strip() for ext in file_pattern.split(',')]
                    matching_files = self.s3_helper.search_files(term, extensions)
                else:
                    matching_files = self.s3_helper.search_files(term)
                
                # Add files that aren't already in the list
                for file in matching_files:
                    if not any(existing['key'] == file['key'] for existing in all_matching_files):
                        all_matching_files.append(file)
            
            if not all_matching_files:
                return {"response": "No files found matching your query.", "full_prompt": query}
            
            # Limit number of files to process
            files_to_process = all_matching_files[:max_files]
            
            # Gather file contents
            file_contents = []
            for file_info in files_to_process:
                try:
                    content = self.s3_helper.get_file_content(file_info['key'])
                    file_contents.append({
                        'filename': file_info['key'],
                        'content': content,
                        'size': file_info['size'],
                        'last_modified': file_info['last_modified']
                    })
                except Exception as e:
                    print(f"Error reading file {file_info['key']}: {str(e)}")
                    continue
            
            # Create enhanced prompt
            context_prompt = self._create_context_prompt(query, file_contents)
            
            # Generate response
            response = self.generate_response(context_prompt)
            
            # Return both response and the full prompt that was sent to LLM
            return {"response": response, "full_prompt": context_prompt}
            
        except Exception as e:
            print(f"Error querying S3 files: {str(e)}")
            raise

    def _enhance_prompt_with_s3_context(self, original_prompt, s3_query):
        """Enhance prompt with S3 file context"""
        try:
            # Search for relevant files
            matching_files = self.s3_helper.search_files(s3_query)
            
            if not matching_files:
                return original_prompt
            
            # Get content from up to 3 most relevant files
            context_info = []
            for file_info in matching_files[:3]:
                try:
                    content = self.s3_helper.get_file_content(file_info['key'])
                    context_info.append(f"File: {file_info['key']}\nContent: {content[:1000]}...")
                except Exception as e:
                    continue
            
            if context_info:
                enhanced_prompt = f"""
Context from S3 files:
{chr(10).join(context_info)}

User Question: {original_prompt}

Please answer the question based on the context provided above.
"""
                return enhanced_prompt
            
            return original_prompt
            
        except Exception as e:
            print(f"Error enhancing prompt with S3 context: {str(e)}")
            return original_prompt

    def _create_context_prompt(self, query, file_contents):
        """Create a context-rich prompt for file-based queries"""
        context_sections = []
        
        for file_data in file_contents:
            # Truncate very long content
            content = file_data['content']
            if len(content) > 2000:
                content = content[:2000] + "... [truncated]"
            
            context_sections.append(f"""
File: {file_data['filename']}
Last Modified: {file_data['last_modified']}
Size: {file_data['size']} bytes
Content:
{content}
""")
        
        context_prompt = f"""
You are an AI assistant that can analyze and answer questions about files stored in an S3 bucket. 

User Query: {query}

Relevant Files Found:
{chr(10).join(context_sections)}

Please analyze the content above and provide a comprehensive answer to the user's query. When the user asks about finding files containing specific words or terms:

1. IDENTIFY ALL FILES that contain the search term(s), including partial matches and variations
2. LIST EACH FILE with the specific context where the term appears
3. Include both exact matches and close variations (e.g., "cook" matches "Cook", "Cooke", "cooking", etc.)
4. If asking for specific information, extract and present it clearly from ALL relevant files
5. Provide a complete summary of findings across all files

Focus on being thorough and comprehensive in identifying all matches across the provided files.
"""
        
        return context_prompt

    def list_s3_files(self):
        """List all files in the S3 bucket"""
        try:
            return self.s3_helper.list_files()
        except Exception as e:
            print(f"Error listing S3 files: {str(e)}")
            raise

    def _extract_search_terms(self, query):
        """Extract search terms from natural language queries"""
        import re
        
        # Convert to lowercase for easier matching
        query_lower = query.lower()
        
        # Common patterns for file search queries
        patterns = [
            r'(?:files?|documents?|items?).*?(?:contain|have|with).*?(?:word|term|text)\s+["\']?(\w+)["\']?',
            r'(?:find|search|look for|get).*?files?.*?(?:contain|have|with).*?["\']?(\w+)["\']?',
            r'(?:files?|documents?).*?(?:that|which).*?(?:contain|have|include).*?["\']?(\w+)["\']?',
            r'(?:list|show).*?files?.*?(?:contain|have|with).*?["\']?(\w+)["\']?',
            r'(?:word|term)\s+["\']?(\w+)["\']?',  # Simple "word X" pattern
        ]
        
        extracted_terms = []
        
        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, query_lower)
            extracted_terms.extend(matches)
        
        # Remove duplicates while preserving order
        unique_terms = []
        for term in extracted_terms:
            if term and term not in unique_terms:
                unique_terms.append(term)
        
        # If no patterns matched, try to find quoted words or significant terms
        if not unique_terms:
            # Look for quoted words
            quoted_matches = re.findall(r'["\'](\w+)["\']', query_lower)
            unique_terms.extend(quoted_matches)
            
            # If still no matches, look for significant words (ignore common words)
            if not unique_terms:
                words = re.findall(r'\b\w+\b', query_lower)
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                               'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                               'files', 'file', 'documents', 'document', 'items', 'item', 'list', 'show', 'find', 
                               'search', 'look', 'get', 'contain', 'have', 'include', 'with', 'that', 'which', 
                               'word', 'term', 'text', 'me', 'i', 'you', 'it', 'they', 'them'}
                
                for word in words:
                    if len(word) > 2 and word not in common_words:
                        unique_terms.append(word)
                        break  # Just take the first significant word
        
        # Fallback: if no meaningful terms found, use the original query
        if not unique_terms:
            unique_terms = [query.strip()]
        
        return unique_terms

    def get_file_summary(self, file_key):
        """Get summary of a specific file"""
        try:
            return self.s3_helper.get_file_summary(file_key)
        except Exception as e:
            print(f"Error getting file summary: {str(e)}")
            raise

    def _generate_gemini_response(self, prompt):
        """Generate response using Gemini model"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating Gemini response: {str(e)}")
            raise

    def _generate_openai_response(self, prompt):
        """Generate response using OpenAI model"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating OpenAI response: {str(e)}")
            raise
