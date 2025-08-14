import os
import os
import asyncio
from typing import List, Dict, Any
import google.generativeai as genai
from .s3_helper import S3Helper
from .chunking_service import ChunkingService
from .vector_db_helper import VectorDBHelper
from .hybrid_search_helper import HybridSearchHelper
import logging

# Conditional OpenAI import to avoid initialization issues when using Gemini
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class RAGLLMHelper:
    """
    RAG-enabled LLM Helper that uses ChunkingService for document operations
    and provides context-aware responses
    """
    
    def __init__(self, provider=None):
        # Debug logging for provider resolution
        env_llm_provider = os.environ.get('LLM_PROVIDER')
        env_provider = os.environ.get('PROVIDER')
        print(f"DEBUG: RAGLLMHelper __init__ called with provider={provider}")
        print(f"DEBUG: Environment LLM_PROVIDER={env_llm_provider}, PROVIDER={env_provider}")
        
        self.provider = provider or os.environ.get('LLM_PROVIDER', os.environ.get('PROVIDER', 'gemini'))
        print(f"DEBUG: Resolved provider to: {self.provider}")
        
        logger.info(f"RAGLLMHelper __init__ called with provider={provider}")
        logger.info(f"Environment LLM_PROVIDER={env_llm_provider}, PROVIDER={env_provider}")
        logger.info(f"Resolved provider to: {self.provider}")
        
        self.s3_helper = S3Helper()
        # Use ChunkingService directly instead of MCP layer
        self.chunking_service = ChunkingService()
        self.db_helper = VectorDBHelper()
        # Initialize hybrid search helper
        self.hybrid_search = HybridSearchHelper(provider=self.provider)
        
        logger.info(f"RAGLLMHelper initializing with provider: {self.provider}")
        
        # Initialize LLM based on provider
        if self.provider == 'gemini':
            self.api_key = os.environ.get('GEMINI_API_KEY')
            self.model_name = os.environ.get('MODEL')
            logger.info(f"Initializing RAGLLMHelper with provider: {self.provider}, model: {self.model_name}")
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        elif self.provider == 'openai':
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI library not available. Please install openai package.")
            self.api_key = os.environ.get('OPENAI_API_KEY')
            self.model_name = os.environ.get('MODEL')
            logger.info(f"Initializing RAGLLMHelper with provider: {self.provider}, model: {self.model_name}")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            self.client = OpenAI(api_key=self.api_key)
        else:
            logger.warning(f"Unknown provider {self.provider}, defaulting to gemini")
            # Default to Gemini if provider is unknown
            self.provider = 'gemini'
            self.api_key = os.environ.get('GEMINI_API_KEY')
            self.model_name = os.environ.get('MODEL')
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
    
    async def generate_rag_response(self, query: str, include_sources: bool = True, 
                                  similarity_threshold: float = 0.7, max_chunks: int = 5) -> dict:
        """
        Generate a response using RAG (Retrieval-Augmented Generation)
        """
        try:
            logger.info(f"Generating RAG response for query: {query}")
            
            # Get context using vector database directly
            logger.info(f"Searching for relevant chunks with threshold {similarity_threshold}")
            search_results = self.db_helper.similarity_search(
                query=query,
                limit=max_chunks,
                similarity_threshold=similarity_threshold
            )
            
            logger.info(f"Found {len(search_results)} relevant chunks")
            
            # Format context and sources
            if search_results:
                context_chunks = []
                sources = []
                
                for i, result in enumerate(search_results):
                    chunk_content = result.get('content', '')
                    s3_key = result.get('s3_key', 'Unknown')
                    chunk_index = result.get('chunk_index', 'N/A')
                    similarity = result.get('similarity_score', 0)
                    
                    context_chunks.append(f"Source {i+1}: {chunk_content}")
                    sources.append({
                        's3_key': s3_key,
                        'chunk_index': chunk_index,
                        'similarity_score': similarity,
                        'content_preview': chunk_content[:200] + '...' if len(chunk_content) > 200 else chunk_content
                    })
                
                context = "\n\n".join(context_chunks)
                context_result = {
                    'context': context,
                    'sources': sources,
                    'source_chunks': len(search_results)
                }
            else:
                context_result = {'context': '', 'sources': []}
            
            context = context_result.get('context', '')
            sources = context_result.get('sources', [])
            
            if not context:
                # Fallback to regular LLM response if no context found
                logger.info("No relevant context found, generating regular response")
                response = await self._generate_llm_response(query)
                return {
                    'response': response,
                    'context_used': False,
                    'sources': [],
                    'message': 'No relevant documents found. Response generated without RAG context.'
                }
            
            # Create RAG prompt
            rag_prompt = self._create_rag_prompt(query, context)
            
            # Generate response
            response = await self._generate_llm_response(rag_prompt)
            
            result = {
                'response': response,
                'context_used': True,
                'sources': sources if include_sources else [],
                'context_chunks': context_result.get('source_chunks', 0)
            }
            
            logger.info(f"Generated RAG response using {len(sources)} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            # Fallback to regular LLM response
            try:
                response = await self._generate_llm_response(query)
                return {
                    'response': response,
                    'context_used': False,
                    'sources': [],
                    'error': f'RAG error: {str(e)}. Generated fallback response.'
                }
            except Exception as fallback_error:
                raise Exception(f"Both RAG and fallback failed: RAG={str(e)}, Fallback={str(fallback_error)}")
    
    def index_document(self, s3_key: str, force_reindex: bool = False, chunking_method: str = 'adaptive') -> dict:
        """Index a document for RAG using enhanced adaptive chunking"""
        try:
            # Use ChunkingService directly instead of MCP
            result = self.chunking_service.process_single_file(
                s3_key=s3_key,
                chunking_method=chunking_method,
                source_type='s3'
            )
            return result
        except Exception as e:
            logger.error(f"Error indexing document {s3_key}: {str(e)}")
            raise
    
    def search_documents(self, query: str, limit: int = 10, 
                       min_similarity: float = 0.5, 
                       chunking_method: str = None) -> List[Dict[str, Any]]:
        """Search indexed documents using vector similarity"""
        try:
            return self.vector_db.search_similar_documents(
                query=query, 
                limit=limit, 
                min_similarity=min_similarity,
                chunking_method=chunking_method
            )
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def list_documents(self, limit: int = 50, status: str = None):
        """List all indexed documents with their metadata"""
        try:
            return self.vector_db.get_documents(limit=limit, status=status)
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            return []
    
    async def query_s3_files_rag(self, query: str, auto_index: bool = False, 
                               similarity_threshold: float = 0.7, max_chunks: int = 5) -> dict:
        """
        Query S3 files using RAG approach
        """
        try:
            logger.info(f"Querying S3 files with RAG for: {query}")
            
            # If auto_index is enabled, try to index any new files
            if auto_index:
                await self._auto_index_new_files()
            
            # Generate RAG response
            result = await self.generate_rag_response(
                query=query,
                similarity_threshold=similarity_threshold,
                max_chunks=max_chunks
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying S3 files with RAG: {str(e)}")
            raise
    
    async def _auto_index_new_files(self):
        """Automatically index new files from S3 that aren't in the vector database"""
        try:
            logger.info("Checking for new files to index")
            
            # Get all S3 files
            s3_files = self.s3_helper.list_files()
            
            # Get already indexed documents from the database
            try:
                # Get list of S3 keys that already have documents in the database
                indexed_docs = set()
                
                # Check which files are already in the documents table
                for file_info in s3_files:
                    s3_key = file_info['key']
                    existing_doc_id = self.db_helper.find_existing_document(s3_key, os.path.basename(s3_key))
                    if existing_doc_id:
                        indexed_docs.add(s3_key)
                
                logger.info(f"Found {len(indexed_docs)} already indexed documents")
                
            except Exception as e:
                logger.warning(f"Error checking existing documents: {e}. Assuming no files are indexed.")
                indexed_docs = set()
            
            # Find new files
            new_files = [f for f in s3_files if f['key'] not in indexed_docs]
            
            if new_files:
                logger.info(f"Found {len(new_files)} new files to index")
                
                # Index up to 2 new files at a time to avoid overwhelming the system
                for file_info in new_files[:2]:  # Reduced from 5 to 2 for safety
                    try:
                        # Check one more time before indexing to avoid race conditions
                        s3_key = file_info['key']
                        existing_doc_id = self.db_helper.find_existing_document(s3_key, os.path.basename(s3_key))
                        if existing_doc_id:
                            logger.info(f"File {s3_key} was already indexed by another process, skipping")
                            continue
                            
                        result = self.index_document(file_info['key'])
                        logger.info(f"Successfully indexed: {file_info['key']} (Document ID: {result.get('document_id', 'unknown')})")
                    except Exception as e:
                        logger.error(f"Failed to index {file_info['key']}: {str(e)}")
            else:
                logger.info("No new files found to index")
                
        except Exception as e:
            logger.error(f"Error in auto-indexing: {str(e)}")
            # Don't raise exception as this is a background operation
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create a prompt that includes retrieved context"""
        return f"""You are an AI assistant that answers questions based on the provided context from documents stored in an S3 bucket.

Context from relevant documents:
{context}

User Question: {query}

Instructions:
1. Answer the question based primarily on the provided context
2. If the context doesn't contain enough information to fully answer the question, say so
3. Cite specific sources when referencing information from the context
4. If you need to make inferences beyond the context, clearly indicate that
5. Be concise but comprehensive in your response

Answer:"""
    
    async def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using the configured LLM provider"""
        if self.provider == 'gemini':
            return await self._generate_gemini_response(prompt)
        elif self.provider == 'openai':
            return await self._generate_openai_response(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    async def _generate_gemini_response(self, prompt: str) -> str:
        """Generate response using Gemini model"""
        try:
            # Run in thread pool since Gemini is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.model.generate_content(prompt)
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            raise
    
    async def _generate_openai_response(self, prompt: str) -> str:
        """Generate response using OpenAI model"""
        try:
            # Run in thread pool since OpenAI client might not be fully async
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating OpenAI response: {str(e)}")
            raise
    
    # Backward compatibility methods for existing API
    def generate_response(self, prompt, include_s3_context=False, s3_query=None):
        """Legacy method for backward compatibility"""
        if include_s3_context and s3_query:
            # Use RAG approach
            result = asyncio.run(self.query_s3_files_rag(s3_query))
            return result.get('response', 'Error generating response')
        else:
            # Use regular LLM response
            result = asyncio.run(self._generate_llm_response(prompt))
            return result
    
    def query_s3_files(self, query, file_pattern=None, max_files=5):
        """Legacy method for backward compatibility - now uses hybrid search for better results"""
        try:
            # Use hybrid search for enhanced results
            result = asyncio.run(self.hybrid_search.hybrid_search_and_answer(
                query=query,
                max_results=max_files if max_files else 10,
                semantic_threshold=0.6
            ))
            
            # Format response to match old format while providing enhanced information
            return {
                'response': result.get('response', ''),
                'full_prompt': query,
                'sources': result.get('sources', []),
                'context_used': result.get('context_used', False),
                'search_strategy': result.get('search_strategy', 'hybrid'),
                'query_analysis': result.get('query_analysis', {}),
                'search_stats': result.get('search_stats', {})
            }
        except Exception as e:
            logger.error(f"Error in hybrid search query: {str(e)}")
            return {
                'response': f"Error: {str(e)}",
                'full_prompt': query,
                'sources': [],
                'context_used': False,
                'search_strategy': 'error',
                'query_analysis': {},
                'search_stats': {}
            }

    async def query_s3_files_hybrid(self, query: str, max_results: int = 10, 
                                   semantic_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Query S3 files using hybrid search approach (keyword + semantic)
        """
        try:
            logger.info(f"Querying S3 files with hybrid search for: {query}")
            
            # Use hybrid search helper
            result = await self.hybrid_search.hybrid_search_and_answer(
                query=query,
                max_results=max_results,
                semantic_threshold=semantic_threshold
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying S3 files with hybrid search: {str(e)}")
            raise
