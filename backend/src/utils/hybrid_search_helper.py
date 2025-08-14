import logging
import re
import json
from typing import List, Dict, Any, Set, Union
import google.generativeai as genai
from .vector_db_helper import VectorDBHelper
import os

logger = logging.getLogger(__name__)

class HybridSearchHelper:
    """
    Hybrid Search Helper that combines keyword-based search with semantic search
    for enhanced document retrieval and query answering.
    """
    
    def __init__(self, provider=None):
        self.provider = provider or os.environ.get('LLM_PROVIDER', os.environ.get('PROVIDER', 'gemini'))
        self.db_helper = VectorDBHelper()
        
        # Initialize LLM for query parsing and final answer generation
        if self.provider == 'gemini':
            self.api_key = os.environ.get('GEMINI_API_KEY')
            self.model_name = os.environ.get('MODEL', 'gemini-1.5-flash')
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        else:
            # For now, only supporting Gemini
            logger.warning(f"Provider {self.provider} not fully supported for hybrid search, defaulting to Gemini")
            self.provider = 'gemini'
            self.api_key = os.environ.get('GEMINI_API_KEY')
            self.model_name = os.environ.get('MODEL', 'gemini-1.5-flash')
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
    
    async def parse_query_for_keywords(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to parse the query and extract specific search keywords/terms
        """
        try:
            logger.info(f"Parsing query for keywords: {query}")
            
            prompt = f"""
            Analyze the following user query and extract specific search keywords and terms that would be most useful for finding relevant documents.

            User Query: "{query}"

            Please provide:
            1. Primary keywords (most important terms)
            2. Secondary keywords (supporting terms)
            3. Entities (names, places, organizations, specific terms)
            4. Question type (factual, procedural, comparative, etc.)

            Return your response in the following JSON format:
            {{
                "primary_keywords": ["keyword1", "keyword2"],
                "secondary_keywords": ["term1", "term2"],
                "entities": ["entity1", "entity2"],
                "question_type": "factual|procedural|comparative|other",
                "search_intent": "brief description of what the user is looking for"
            }}

            Focus on extracting actual searchable terms that would appear in documents, not generic words.
            """
            
            # Add timeout and error handling for LLM call
            try:
                import asyncio
                # Use asyncio.wait_for to add timeout to the blocking call
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.model.generate_content, prompt
                    ),
                    timeout=15.0  # 15 second timeout for keyword parsing
                )
            except asyncio.TimeoutError:
                logger.error(f"LLM call timed out for query parsing: {query}")
                return self._fallback_keyword_extraction(query)
            except Exception as llm_error:
                logger.error(f"LLM call failed for query parsing: {query}, error: {str(llm_error)}")
                return self._fallback_keyword_extraction(query)
            
            # Parse the JSON response
            try:
                # Extract JSON from response text
                response_text = response.text.strip()
                # Handle markdown code blocks
                if '```json' in response_text:
                    json_start = response_text.find('```json') + 7
                    json_end = response_text.find('```', json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif '```' in response_text:
                    json_start = response_text.find('```') + 3
                    json_end = response_text.find('```', json_start)
                    response_text = response_text[json_start:json_end].strip()
                
                parsed = json.loads(response_text)
                logger.info(f"Extracted keywords: {parsed}")
                return parsed
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse LLM response as JSON: {e}")
                # Fallback: extract keywords manually
                return self._fallback_keyword_extraction(query)
                
        except Exception as e:
            logger.error(f"Error parsing query for keywords: {str(e)}")
            # Fallback: simple keyword extraction
            return self._fallback_keyword_extraction(query)
    
    def _fallback_keyword_extraction(self, query: str) -> Dict[str, Any]:
        """
        Fallback method for keyword extraction using simple text processing
        """
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'what', 'where', 'when', 'why', 'how', 'who', 'which', 'that', 'this', 'these',
            'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
            'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
            'their', 'theirs', 'themselves'
        }
        
        # Clean and tokenize query
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Simple heuristics for categorization
        primary_keywords = keywords[:3]  # First 3 non-stop words
        secondary_keywords = keywords[3:6]  # Next 3 words
        
        # Simple entity detection (capitalized words in original query)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        return {
            "primary_keywords": primary_keywords,
            "secondary_keywords": secondary_keywords,
            "entities": entities,
            "question_type": "other",
            "search_intent": f"Find information related to: {', '.join(primary_keywords)}"
        }
    
    def keyword_search(self, keywords: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search using PostgreSQL full-text search
        """
        try:
            logger.info(f"Performing keyword search for: {keywords}")
            
            if not keywords:
                return []
            
            conn = self.db_helper._get_connection()
            cursor = conn.cursor()
            
            # Build full-text search query - handle phrases and single words properly
            processed_terms = []
            for keyword in keywords:
                # For phrases (multiple words), use plainto_tsquery which handles phrases better
                # For single words, we can use them directly
                escaped_keyword = keyword.replace("'", "''")  # Escape single quotes for SQL
                processed_terms.append(f"plainto_tsquery('english', '{escaped_keyword}')") 
            
            # Combine with OR logic
            search_condition = ' OR '.join([f"to_tsvector('english', dc.content) @@ {term}" for term in processed_terms])
            
            query = f"""
                SELECT 
                    dc.id,
                    dc.document_id,
                    d.s3_key,
                    dc.chunk_index,
                    dc.content,
                    dc.metadata,
                    dc.chunking_method,
                    GREATEST({', '.join([f"ts_rank(to_tsvector('english', dc.content), {term})" for term in processed_terms])}) as rank
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                WHERE {search_condition}
                ORDER BY rank DESC
                LIMIT %s
            """
            
            cursor.execute(query, [limit])
            results = cursor.fetchall()
            
            # Format results
            formatted_results = []
            for row in results:
                formatted_results.append({
                    'id': row[0],
                    'document_id': row[1],
                    's3_key': row[2],
                    'chunk_index': row[3],
                    'content': row[4],
                    'metadata': json.loads(row[5]) if isinstance(row[5], str) else row[5],
                    'chunking_method': row[6],
                    'keyword_rank': float(row[7]) if row[7] else 0.0,
                    'search_type': 'keyword'
                })
            
            logger.info(f"Keyword search found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in keyword search: {str(e)}")
            return []
    
    def semantic_search(self, query: str, limit: int = 10, similarity_threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Perform semantic search using vector embeddings
        """
        try:
            logger.info(f"Performing semantic search for: {query}")
            
            # Use the existing similarity_search method which generates its own embedding
            results = self.db_helper.similarity_search(
                query=query,
                limit=limit,
                similarity_threshold=similarity_threshold
            )
            
            # Add search type marker
            for result in results:
                result['search_type'] = 'semantic'
            
            logger.info(f"Semantic search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def merge_and_rank_results(self, keyword_results: List[Dict], semantic_results: List[Dict], 
                              max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Merge and rank results from keyword and semantic search
        """
        try:
            # Create a dictionary to merge duplicate documents
            merged_results = {}
            
            # Process keyword results
            for result in keyword_results:
                key = f"{result['document_id']}_{result['chunk_index']}"
                if key not in merged_results:
                    merged_results[key] = result.copy()
                    merged_results[key]['combined_score'] = result.get('keyword_rank', 0) * 0.5
                    merged_results[key]['keyword_found'] = True
                    merged_results[key]['semantic_found'] = False
                else:
                    # Already exists from semantic search
                    merged_results[key]['keyword_found'] = True
                    merged_results[key]['combined_score'] += result.get('keyword_rank', 0) * 0.5
            
            # Process semantic results
            for result in semantic_results:
                key = f"{result['document_id']}_{result['chunk_index']}"
                if key not in merged_results:
                    merged_results[key] = result.copy()
                    merged_results[key]['combined_score'] = result.get('similarity_score', 0) * 0.5
                    merged_results[key]['keyword_found'] = False
                    merged_results[key]['semantic_found'] = True
                else:
                    # Already exists from keyword search
                    merged_results[key]['semantic_found'] = True
                    merged_results[key]['combined_score'] += result.get('similarity_score', 0) * 0.5
            
            # Convert to list and sort by combined score
            final_results = list(merged_results.values())
            
            # Sort by combined score (highest first)
            final_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
            
            # Limit results
            final_results = final_results[:max_results]
            
            logger.info(f"Merged results: {len(final_results)} chunks from {len(keyword_results)} keyword + {len(semantic_results)} semantic results")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error merging and ranking results: {str(e)}")
            # Fallback: just combine and limit
            all_results = keyword_results + semantic_results
            return all_results[:max_results]
    
    async def generate_final_answer(self, query: str, relevant_chunks: List[Dict], 
                                  query_analysis: Dict) -> Dict[str, Any]:
        """
        Use LLM to analyze the retrieved document chunks and generate a comprehensive answer
        """
        try:
            logger.info(f"Generating final answer for query: {query}")
            
            if not relevant_chunks:
                return {
                    'response': "I couldn't find any relevant information in the documents to answer your query.",
                    'sources': [],
                    'context_used': False,
                    'search_strategy': 'hybrid',
                    'query_analysis': query_analysis
                }
            
            # Prepare context from relevant chunks
            context_parts = []
            sources = []
            
            for i, chunk in enumerate(relevant_chunks):
                context_parts.append(f"[Source {i+1}] From {chunk.get('s3_key', 'Unknown')}, Chunk {chunk.get('chunk_index', 0)}:\n{chunk.get('content', '')}")
                
                sources.append({
                    'chunk_index': chunk.get('chunk_index', 0),
                    's3_key': chunk.get('s3_key', 'Unknown'),
                    'similarity_score': chunk.get('similarity_score', 0),
                    'keyword_rank': chunk.get('keyword_rank', 0),
                    'combined_score': chunk.get('combined_score', 0),
                    'search_type': chunk.get('search_type', 'hybrid'),
                    'content_preview': chunk.get('content', '')[:200] + '...' if len(chunk.get('content', '')) > 200 else chunk.get('content', '')
                })
            
            context_text = '\n\n'.join(context_parts)
            
            # Create comprehensive prompt for final answer
            prompt = f"""
            You are an AI assistant that provides accurate and helpful answers based on document content. 
            
            User Query: "{query}"
            
            Query Analysis:
            - Primary Keywords: {', '.join(query_analysis.get('primary_keywords', []))}
            - Secondary Keywords: {', '.join(query_analysis.get('secondary_keywords', []))}
            - Entities: {', '.join(query_analysis.get('entities', []))}
            - Question Type: {query_analysis.get('question_type', 'unknown')}
            - Search Intent: {query_analysis.get('search_intent', 'general information')}
            
            Relevant Document Content:
            {context_text}
            
            Instructions:
            1. Carefully analyze the provided document content
            2. Answer the user's query based ONLY on the information in the documents
            3. If the documents don't contain enough information to fully answer the query, say so
            4. Cite specific sources when providing information (e.g., "According to Source 1...")
            5. Be comprehensive but concise
            6. If multiple sources provide different perspectives, mention both
            
            Please provide a detailed and accurate answer to the user's query:
            """
            
            # Add timeout and error handling for LLM call
            try:
                import asyncio
                # Use asyncio.wait_for to add timeout to the blocking call
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, self.model.generate_content, prompt
                    ),
                    timeout=30.0  # 30 second timeout
                )
                answer = response.text.strip()
            except asyncio.TimeoutError:
                logger.error(f"LLM call timed out for query: {query}")
                answer = "I found relevant information in the documents, but the response generation timed out. Please try again."
            except Exception as llm_error:
                logger.error(f"LLM call failed for query: {query}, error: {str(llm_error)}")
                answer = f"I found relevant information in the documents, but encountered an error generating the response: {str(llm_error)}"
            
            return {
                'response': answer,
                'sources': sources,
                'context_used': True,
                'context_chunks': len(relevant_chunks),
                'search_strategy': 'hybrid',
                'query_analysis': query_analysis
            }
            
        except Exception as e:
            logger.error(f"Error generating final answer: {str(e)}")
            return {
                'response': f"I found relevant information but encountered an error generating the response: {str(e)}",
                'sources': sources if 'sources' in locals() else [],
                'context_used': True,
                'search_strategy': 'hybrid',
                'query_analysis': query_analysis
            }
    
    async def hybrid_search_and_answer(self, query: str, max_results: int = 10, 
                                     semantic_threshold: float = 0.6) -> Dict[str, Any]:
        """
        Main method that performs hybrid search and generates answer
        """
        try:
            logger.info(f"Starting hybrid search for query: {query}")
            
            # Step 1: Parse query for keywords using LLM
            query_analysis = await self.parse_query_for_keywords(query)
            
            # Step 2: Extract all keywords for search
            all_keywords = (
                query_analysis.get('primary_keywords', []) + 
                query_analysis.get('secondary_keywords', []) + 
                query_analysis.get('entities', [])
            )
            
            # Step 3: Perform keyword search
            keyword_results = self.keyword_search(all_keywords, limit=max_results)
            
            # Step 4: Perform semantic search
            semantic_results = self.semantic_search(query, limit=max_results, similarity_threshold=semantic_threshold)
            
            # Step 5: Merge and rank results
            merged_results = self.merge_and_rank_results(keyword_results, semantic_results, max_results)
            
            # Step 6: Generate final answer using LLM
            final_response = await self.generate_final_answer(query, merged_results, query_analysis)
            
            # Add search statistics
            final_response['search_stats'] = {
                'keyword_results_count': len(keyword_results),
                'semantic_results_count': len(semantic_results),
                'merged_results_count': len(merged_results),
                'keywords_used': all_keywords
            }
            
            logger.info(f"Hybrid search completed successfully")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in hybrid search and answer: {str(e)}")
            return {
                'response': f"I encountered an error while searching for information: {str(e)}",
                'sources': [],
                'context_used': False,
                'search_strategy': 'hybrid',
                'query_analysis': {},
                'search_stats': {
                    'keyword_results_count': 0,
                    'semantic_results_count': 0,
                    'merged_results_count': 0,
                    'keywords_used': []
                }
            }
