import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import time

try:
    import openai
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class RAGAHelper:
    """
    RAGA (Retrieval-Augmented Generation with Assessment) Helper
    
    Provides advanced RAG capabilities with:
    - Multi-source retrieval
    - Answer quality assessment
    - Citation tracking
    - Response validation
    - Performance metrics
    """
    
    def __init__(self, vector_db_helper, s3_helper):
        self.vector_db_helper = vector_db_helper
        self.s3_helper = s3_helper
        
        # LLM Configuration
        self.llm_provider = os.environ.get('LLM_PROVIDER', 'gemini')
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        self.azure_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        self.azure_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
        self.azure_api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        
        # Initialize LLM clients
        self._init_llm_clients()
        
        # RAGA Configuration
        self.max_retrieval_chunks = int(os.environ.get('RAGA_MAX_CHUNKS', 10))
        self.similarity_threshold = float(os.environ.get('RAGA_SIMILARITY_THRESHOLD', 0.75))
        self.answer_confidence_threshold = float(os.environ.get('RAGA_CONFIDENCE_THRESHOLD', 0.7))
        self.enable_assessment = os.environ.get('RAGA_ENABLE_ASSESSMENT', 'true').lower() == 'true'
        self.enable_citations = os.environ.get('RAGA_ENABLE_CITATIONS', 'true').lower() == 'true'
        
        logger.info(f"RAGA Helper initialized with provider: {self.llm_provider}")
    
    def _init_llm_clients(self):
        """Initialize LLM clients"""
        self.llm_client = None
        self.gemini_model = None
        
        try:
            if self.llm_provider == 'azure' and self.azure_api_key and OPENAI_AVAILABLE:
                self.llm_client = AzureOpenAI(
                    api_key=self.azure_api_key,
                    api_version=self.azure_api_version,
                    azure_endpoint=self.azure_endpoint
                )
                logger.info("Azure OpenAI client initialized")
            elif self.llm_provider == 'openai' and self.openai_api_key and OPENAI_AVAILABLE:
                openai.api_key = self.openai_api_key
                self.llm_client = openai
                logger.info("OpenAI client initialized")
            elif self.llm_provider == 'gemini' and self.gemini_api_key and GEMINI_AVAILABLE:
                genai.configure(api_key=self.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini client initialized")
            else:
                logger.warning(f"LLM provider {self.llm_provider} not available or not configured")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
    
    async def raga_query(self, 
                        query: str, 
                        max_chunks: int = None,
                        similarity_threshold: float = None,
                        enable_assessment: bool = None,
                        file_filters: List[str] = None) -> Dict[str, Any]:
        """
        Perform a comprehensive RAGA query with retrieval, generation, and assessment
        
        Args:
            query: User's question
            max_chunks: Maximum chunks to retrieve
            similarity_threshold: Minimum similarity for chunks
            enable_assessment: Whether to assess answer quality
            file_filters: List of file patterns to filter sources
        
        Returns:
            Dict with answer, sources, assessment, and metadata
        """
        start_time = time.time()
        
        # Use instance defaults if not provided
        max_chunks = max_chunks or self.max_retrieval_chunks
        similarity_threshold = similarity_threshold or self.similarity_threshold
        enable_assessment = enable_assessment if enable_assessment is not None else self.enable_assessment
        
        try:
            # Step 1: Retrieval Phase
            logger.info(f"Starting RAGA query: {query[:100]}...")
            retrieval_results = await self._enhanced_retrieval(
                query, max_chunks, similarity_threshold, file_filters
            )
            
            if not retrieval_results['chunks']:
                return self._create_no_results_response(query, retrieval_results, similarity_threshold)
            
            # Step 2: Generation Phase
            generation_results = await self._enhanced_generation(
                query, retrieval_results['chunks']
            )
            
            # Step 3: Assessment Phase (optional)
            assessment_results = None
            if enable_assessment:
                assessment_results = await self._assess_answer_quality(
                    query, generation_results['answer'], retrieval_results['chunks']
                )
            
            # Step 4: Compile comprehensive response
            end_time = time.time()
            processing_time = end_time - start_time
            
            return self._compile_raga_response(
                query=query,
                retrieval_results=retrieval_results,
                generation_results=generation_results,
                assessment_results=assessment_results,
                processing_time=processing_time,
                similarity_threshold=similarity_threshold
            )
            
        except Exception as e:
            logger.error(f"RAGA query failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'query': query,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _enhanced_retrieval(self, 
                                query: str, 
                                max_chunks: int, 
                                similarity_threshold: float,
                                file_filters: List[str] = None) -> Dict[str, Any]:
        """Enhanced retrieval with multiple strategies and quality filtering"""
        
        # Strategy 1: Direct similarity search
        primary_chunks = self.vector_db_helper.similarity_search(
            query=query,
            limit=max_chunks,
            similarity_threshold=similarity_threshold
        )
        
        # Strategy 2: Query expansion for better retrieval
        expanded_queries = await self._expand_query(query)
        expanded_chunks = []
        
        for expanded_query in expanded_queries:
            additional_chunks = self.vector_db_helper.similarity_search(
                query=expanded_query,
                limit=max_chunks // 2,
                similarity_threshold=similarity_threshold * 0.9  # Slightly lower threshold
            )
            expanded_chunks.extend(additional_chunks)
        
        # Combine and deduplicate chunks
        all_chunks = primary_chunks + expanded_chunks
        unique_chunks = self._deduplicate_chunks(all_chunks)
        
        # Apply file filters if provided
        if file_filters:
            unique_chunks = self._apply_file_filters(unique_chunks, file_filters)
        
        # Rank and select best chunks
        ranked_chunks = self._rank_chunks_by_relevance(query, unique_chunks)
        final_chunks = ranked_chunks[:max_chunks]
        
        return {
            'chunks': final_chunks,
            'total_found': len(unique_chunks),
            'primary_count': len(primary_chunks),
            'expanded_count': len(expanded_chunks),
            'strategies_used': ['similarity_search', 'query_expansion'],
            'file_filters_applied': file_filters or []
        }
    
    async def _expand_query(self, query: str) -> List[str]:
        """Generate expanded queries for better retrieval coverage"""
        if not self._llm_available():
            return []
        
        expansion_prompt = f"""
        Given this user query, generate 2-3 alternative phrasings or related questions that would help find relevant information:
        
        Original query: "{query}"
        
        Generate variations that:
        1. Use different terminology/synonyms
        2. Ask about related concepts
        3. Use different question structures
        
        Return only the alternative queries, one per line, without numbering or explanations.
        """
        
        try:
            response = await self._call_llm(expansion_prompt, max_tokens=200)
            if response:
                expanded_queries = [q.strip() for q in response.split('\n') if q.strip() and q.strip() != query]
                return expanded_queries[:3]  # Limit to 3 expansions
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        
        return []
    
    async def _enhanced_generation(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive answer with citations and confidence scoring"""
        
        # Prepare context from chunks
        context = self._prepare_context_with_citations(chunks)
        
        # Create enhanced prompt
        enhanced_prompt = self._create_generation_prompt(query, context, chunks)
        
        # Generate answer
        start_time = time.time()
        answer = await self._call_llm(enhanced_prompt, max_tokens=1500)
        generation_time = time.time() - start_time
        
        if not answer:
            return {
                'answer': "I apologize, but I couldn't generate a response based on the available information.",
                'confidence': 0.0,
                'generation_time': generation_time,
                'prompt_used': enhanced_prompt
            }
        
        # Extract confidence if available in answer
        confidence = self._extract_confidence_from_answer(answer)
        
        return {
            'answer': answer,
            'confidence': confidence,
            'generation_time': generation_time,
            'prompt_used': enhanced_prompt,
            'context_chunks_used': len(chunks)
        }
    
    async def _assess_answer_quality(self, 
                                   query: str, 
                                   answer: str, 
                                   source_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess the quality and reliability of the generated answer"""
        
        if not self._llm_available():
            return {'assessment_available': False}
        
        assessment_prompt = f"""
        Assess the quality of this answer to the given question based on the provided source material.
        
        Question: "{query}"
        
        Answer: "{answer}"
        
        Evaluate the answer on these criteria (score 1-10 for each):
        1. Accuracy: How well does the answer align with the source material?
        2. Completeness: How thoroughly does the answer address the question?
        3. Clarity: How clear and understandable is the answer?
        4. Relevance: How relevant is the answer to the specific question?
        5. Citation Quality: How well are sources referenced?
        
        Provide scores and brief explanations in this JSON format:
        {{
            "accuracy": {{"score": X, "explanation": "..."}},
            "completeness": {{"score": X, "explanation": "..."}},
            "clarity": {{"score": X, "explanation": "..."}},
            "relevance": {{"score": X, "explanation": "..."}},
            "citation_quality": {{"score": X, "explanation": "..."}},
            "overall_score": X,
            "overall_assessment": "...",
            "recommendations": ["..."]
        }}
        """
        
        try:
            assessment_response = await self._call_llm(assessment_prompt, max_tokens=800)
            if assessment_response:
                # Try to parse JSON response
                try:
                    assessment_data = json.loads(assessment_response)
                    assessment_data['assessment_available'] = True
                    return assessment_data
                except json.JSONDecodeError:
                    # Fallback to text-based assessment
                    return {
                        'assessment_available': True,
                        'text_assessment': assessment_response,
                        'overall_score': 7.0  # Default score
                    }
        except Exception as e:
            logger.warning(f"Answer assessment failed: {e}")
        
        return {'assessment_available': False}
    
    def _prepare_context_with_citations(self, chunks: List[Dict[str, Any]]) -> str:
        """Prepare context string with citation markers"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get('content', '')
            s3_key = chunk.get('s3_key') or 'unknown'
            chunking_method = chunk.get('chunking_method', 'unknown')
            
            citation_marker = f"[Source {i}: {s3_key}]"
            context_part = f"{citation_marker}\n{content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_generation_prompt(self, query: str, context: str, chunks: List[Dict[str, Any]]) -> str:
        """Create an enhanced prompt for answer generation"""
        
        source_files = list(set(chunk.get('s3_key', 'unknown') for chunk in chunks if chunk.get('s3_key') is not None))
        
        prompt = f"""
You are an expert AI assistant providing accurate, well-sourced answers based on document analysis.

QUESTION: {query}

AVAILABLE CONTEXT:
{context}

INSTRUCTIONS:
1. Provide a comprehensive, accurate answer based ONLY on the information in the provided context
2. Include specific citations using the [Source X] format when referencing information
3. If information is insufficient, clearly state what aspects cannot be answered
4. Organize your response clearly with proper structure
5. Be objective and factual
6. At the end, provide a confidence score (0.0-1.0) for your answer

SOURCES CONSULTED: {', '.join(source_files)}

ANSWER:
"""
        return prompt
    
    def _extract_confidence_from_answer(self, answer: str) -> float:
        """Extract confidence score from answer if present"""
        import re
        
        # Look for confidence patterns
        confidence_patterns = [
            r'confidence[:\s]*([0-9]*\.?[0-9]+)',
            r'confidence score[:\s]*([0-9]*\.?[0-9]+)',
            r'score[:\s]*([0-9]*\.?[0-9]+)'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, answer.lower())
            if match:
                try:
                    score = float(match.group(1))
                    return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
                except ValueError:
                    continue
        
        # Default confidence based on answer quality indicators
        quality_indicators = ['specifically', 'according to', 'source', 'document', 'evidence']
        quality_score = sum(1 for indicator in quality_indicators if indicator in answer.lower())
        
        return min(0.5 + (quality_score * 0.1), 0.9)  # Base 0.5, up to 0.9
    
    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate chunks based on content similarity"""
        unique_chunks = []
        seen_content = set()
        
        for chunk in chunks:
            content = chunk.get('content', '')
            content_hash = hash(content[:200])  # Use first 200 chars for deduplication
            
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def _apply_file_filters(self, chunks: List[Dict[str, Any]], file_filters: List[str]) -> List[Dict[str, Any]]:
        """Filter chunks based on file patterns"""
        if not file_filters:
            return chunks
        
        filtered_chunks = []
        for chunk in chunks:
            s3_key = chunk.get('s3_key', '').lower()
            if any(pattern.lower() in s3_key for pattern in file_filters):
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def _rank_chunks_by_relevance(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank chunks by relevance to query"""
        # Simple keyword-based ranking for now
        query_words = set(query.lower().split())
        
        def relevance_score(chunk):
            content = chunk.get('content', '').lower()
            content_words = set(content.split())
            overlap = len(query_words.intersection(content_words))
            similarity = chunk.get('similarity_score', 0.0)
            return overlap + similarity
        
        return sorted(chunks, key=relevance_score, reverse=True)
    
    def _compile_raga_response(self, 
                              query: str,
                              retrieval_results: Dict[str, Any],
                              generation_results: Dict[str, Any],
                              assessment_results: Optional[Dict[str, Any]],
                              processing_time: float,
                              similarity_threshold: float = None) -> Dict[str, Any]:
        """Compile comprehensive RAGA response"""
        
        chunks = retrieval_results['chunks']
        
        # Use provided threshold or fall back to instance default
        threshold_used = similarity_threshold or self.similarity_threshold
        
        # Extract source information
        sources = []
        for i, chunk in enumerate(chunks, 1):
            sources.append({
                'id': i,
                's3_key': chunk.get('s3_key', 'unknown'),
                'chunk_index': chunk.get('chunk_index', 0),
                'similarity_score': chunk.get('similarity_score', 0.0),
                'chunking_method': chunk.get('chunking_method', 'unknown'),
                'content_preview': chunk.get('content', '')[:200] + '...' if len(chunk.get('content', '')) > 200 else chunk.get('content', '')
            })
        
        response = {
            'success': True,
            'query': query,
            'answer': generation_results['answer'],
            'confidence': generation_results.get('confidence', 0.5),
            'sources': sources,
            'metadata': {
                'processing_time': processing_time,
                'generation_time': generation_results.get('generation_time', 0),
                'total_chunks_found': retrieval_results['total_found'],
                'chunks_used': len(chunks),
                'retrieval_strategies': retrieval_results['strategies_used'],
                'llm_provider': self.llm_provider,
                'similarity_threshold': threshold_used,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Add assessment if available
        if assessment_results and assessment_results.get('assessment_available'):
            response['quality_assessment'] = assessment_results
        
        return response
    
    def _create_no_results_response(self, query: str, retrieval_results: Dict[str, Any], similarity_threshold: float = None) -> Dict[str, Any]:
        """Create response when no relevant chunks are found"""
        threshold_used = similarity_threshold or self.similarity_threshold
        return {
            'success': True,
            'query': query,
            'answer': "I couldn't find relevant information in the available documents to answer your question. Please try rephrasing your query or check if the relevant documents have been indexed.",
            'confidence': 0.0,
            'sources': [],
            'metadata': {
                'total_chunks_found': retrieval_results['total_found'],
                'chunks_used': 0,
                'retrieval_strategies': retrieval_results['strategies_used'],
                'similarity_threshold': threshold_used,
                'timestamp': datetime.now().isoformat(),
                'no_results_reason': 'No chunks met similarity threshold'
            }
        }
    
    async def _call_llm(self, prompt: str, max_tokens: int = 1000) -> Optional[str]:
        """Call the configured LLM with the given prompt"""
        if not self._llm_available():
            return None
        
        try:
            if self.llm_provider == 'gemini' and self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            elif self.llm_provider in ['openai', 'azure'] and self.llm_client:
                if self.llm_provider == 'azure':
                    response = self.llm_client.chat.completions.create(
                        model="gpt-35-turbo",  # Adjust model name as needed
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=0.1
                    )
                    return response.choices[0].message.content
                else:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=max_tokens,
                        temperature=0.1
                    )
                    return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    def _llm_available(self) -> bool:
        """Check if LLM is available and configured"""
        if self.llm_provider == 'gemini':
            return self.gemini_model is not None
        elif self.llm_provider in ['openai', 'azure']:
            return self.llm_client is not None
        return False
    
    async def get_raga_stats(self) -> Dict[str, Any]:
        """Get RAGA system statistics"""
        try:
            # Get vector database stats
            vector_stats = self.vector_db_helper.get_stats()
            
            # Get S3 stats
            s3_files = self.s3_helper.list_objects()
            
            return {
                'vector_database': vector_stats,
                'source_files': {
                    'total_files': len(s3_files),
                    'indexed_files': vector_stats.get('indexed_files', 0)
                },
                'raga_config': {
                    'llm_provider': self.llm_provider,
                    'max_retrieval_chunks': self.max_retrieval_chunks,
                    'similarity_threshold': self.similarity_threshold,
                    'assessment_enabled': self.enable_assessment,
                    'citations_enabled': self.enable_citations
                },
                'capabilities': {
                    'llm_available': self._llm_available(),
                    'query_expansion': self._llm_available(),
                    'answer_assessment': self._llm_available(),
                    'multi_strategy_retrieval': True
                }
            }
        except Exception as e:
            logger.error(f"Failed to get RAGA stats: {e}")
            return {'error': str(e)}
