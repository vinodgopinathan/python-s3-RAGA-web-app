import os
import re
from typing import List, Dict, Any, Optional
import logging
import numpy as np

# Import for semantic chunking
try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

# Import for agentic chunking
try:
    import google.generativeai as genai
    AGENTIC_CHUNKING_AVAILABLE = True
except ImportError:
    AGENTIC_CHUNKING_AVAILABLE = False

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = int(os.environ.get('CHUNK_SIZE', 1000))
        self.chunk_overlap = int(os.environ.get('CHUNK_OVERLAP', 200))
        self.max_context_length = int(os.environ.get('MAX_CONTEXT_LENGTH', 4000))
        self.use_recursive_chunking = os.environ.get('USE_RECURSIVE_CHUNKING', 'true').lower() == 'true'
        
        # Chunking method selection
        self.chunking_method = os.environ.get('CHUNKING_METHOD', 'recursive')  # recursive, semantic, agentic, adaptive
        
        # Semantic chunking threshold
        self.semantic_threshold = float(os.environ.get('SEMANTIC_THRESHOLD', 0.7))
        
        # Initialize semantic model if available
        self.semantic_model = None
        if SEMANTIC_CHUNKING_AVAILABLE and self.chunking_method in ['semantic', 'adaptive']:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Semantic chunking model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
                self.semantic_model = None
        
        # LLM settings for agentic chunking (Gemini only)
        self.llm_provider = os.environ.get('LLM_PROVIDER', 'gemini')  # gemini only
        self.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        
        # Initialize Gemini client for agentic chunking
        if AGENTIC_CHUNKING_AVAILABLE and self.chunking_method in ['agentic', 'adaptive']:
            self._init_llm_clients()
        
        # Recursive chunking separators in order of preference
        self.recursive_separators = [
            "\n\n",      # Double newlines (paragraphs)
            "\n",        # Single newlines
            ". ",        # Sentence endings with space
            "! ",        # Exclamation with space
            "? ",        # Question with space
            "; ",        # Semicolon with space
            ", ",        # Comma with space
            " ",         # Single space (words)
            ""           # Character level (last resort)
        ]
    
    def _init_llm_clients(self):
        """Initialize Gemini client for agentic chunking"""
        try:
            if self.llm_provider == 'gemini' and self.gemini_api_key:
                genai.configure(api_key=self.gemini_api_key)
                logger.info("Gemini client initialized for agentic chunking")
            else:
                logger.info(f"LLM provider {self.llm_provider} configured without initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")
    
    def set_chunking_method(self, method: str):
        """
        Dynamically set the chunking method and reinitialize LLM clients if needed
        """
        self.chunking_method = method.lower()
        
        # Reinitialize LLM clients if switching to agentic chunking
        if AGENTIC_CHUNKING_AVAILABLE and self.chunking_method in ['agentic', 'adaptive']:
            self._init_llm_clients()
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks using the configured chunking method.
        Supports recursive, semantic, agentic, and adaptive chunking.
        """
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # If text is shorter than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [{
                'content': text,
                'chunk_index': 0,
                'metadata': {
                    **(metadata or {}),
                    'chunking_method': 'single_chunk',
                    'char_count': len(text)
                }
            }]
        
        # Choose chunking method - RESPECT USER SELECTION
        method = self.chunking_method.lower()
        
        if method == 'semantic' and self.semantic_model:
            chunks = self._semantic_chunk_text(text, metadata)
        elif method == 'agentic':
            if AGENTIC_CHUNKING_AVAILABLE:
                chunks = self._agentic_chunk_text(text, metadata)
            else:
                logger.warning("Agentic chunking requested but not available, falling back to recursive chunking")
                chunks = self._recursive_chunk_text(text, metadata)
                # Update method for metadata
                method = 'recursive_fallback'
        elif method == 'adaptive':
            chunks = self._adaptive_intelligent_chunk_text(text, metadata)
        elif method == 'recursive':
            chunks = self._recursive_chunk_text(text, metadata)
        elif method == 'sentence':
            chunks = self._sentence_chunk_text(text, metadata)
        else:
            # Default fallback - use recursive chunking
            chunks = self._recursive_chunk_text(text, metadata)
            method = 'recursive'
        
        logger.info(f"Split text into {len(chunks)} chunks using {method} chunking")
        return chunks
    
    def _recursive_chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Recursive text chunking that tries different separators in order of preference
        """
        chunks = []
        
        def _split_text_recursive(text: str, separators: List[str], chunk_index: int = 0) -> List[Dict[str, Any]]:
            """Recursively split text using different separators"""
            
            # If text is small enough, return as chunk
            if len(text) <= self.chunk_size:
                if text.strip():
                    return [{
                        'content': text.strip(),
                        'chunk_index': chunk_index,
                        'metadata': {
                            **(metadata or {}),
                            'chunking_method': 'recursive',
                            'char_count': len(text.strip()),
                            'separator_used': 'none'
                        }
                    }]
                return []
            
            # Try each separator
            for separator in separators:
                if separator in text:
                    # Split by this separator
                    splits = text.split(separator)
                    
                    # If we got meaningful splits, process them
                    if len(splits) > 1:
                        good_splits = []
                        current_chunk = ""
                        
                        for i, split in enumerate(splits):
                            # Reconstruct with separator (except last)
                            if i < len(splits) - 1:
                                split_with_sep = split + separator
                            else:
                                split_with_sep = split
                            
                            # Check if adding this split exceeds chunk size
                            if len(current_chunk + split_with_sep) <= self.chunk_size:
                                current_chunk += split_with_sep
                            else:
                                # Save current chunk if it has content
                                if current_chunk.strip():
                                    good_splits.append(current_chunk.strip())
                                
                                # Start new chunk with overlap if needed
                                if self.chunk_overlap > 0 and good_splits:
                                    overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                                    current_chunk = overlap_text + split_with_sep
                                else:
                                    current_chunk = split_with_sep
                        
                        # Add final chunk
                        if current_chunk.strip():
                            good_splits.append(current_chunk.strip())
                        
                        # Process each split recursively if still too large
                        result_chunks = []
                        for i, chunk_text in enumerate(good_splits):
                            if len(chunk_text) <= self.chunk_size:
                                result_chunks.append({
                                    'content': chunk_text,
                                    'chunk_index': chunk_index + i,
                                    'metadata': {
                                        **(metadata or {}),
                                        'chunking_method': 'recursive',
                                        'char_count': len(chunk_text),
                                        'separator_used': repr(separator)
                                    }
                                })
                            else:
                                # Recursively split large chunks with remaining separators
                                remaining_separators = separators[separators.index(separator) + 1:]
                                sub_chunks = _split_text_recursive(chunk_text, remaining_separators, chunk_index + i)
                                result_chunks.extend(sub_chunks)
                        
                        return result_chunks
            
            # If no separator worked, fall back to character-level splitting
            return self._character_level_split(text, metadata, chunk_index)
        
        return _split_text_recursive(text, self.recursive_separators, 0)
    
    def _semantic_chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Semantic chunking using sentence embeddings to find natural boundaries
        """
        if not self.semantic_model:
            logger.warning("Semantic model not available, falling back to recursive chunking")
            return self._recursive_chunk_text(text, metadata)
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [{
                'content': text,
                'chunk_index': 0,
                'metadata': {
                    **(metadata or {}),
                    'chunking_method': 'semantic_single',
                    'char_count': len(text)
                }
            }]
        
        try:
            # Generate embeddings for sentences
            embeddings = self.semantic_model.encode(sentences)
            
            # Calculate similarity between consecutive sentences
            similarities = []
            for i in range(len(embeddings) - 1):
                similarity = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(similarity)
            
            # Find chunk boundaries where similarity drops below threshold
            chunks = []
            current_chunk_sentences = [sentences[0]]
            chunk_index = 0
            
            for i, similarity in enumerate(similarities):
                # Add next sentence to current chunk
                current_chunk_sentences.append(sentences[i + 1])
                
                # Check if we should create a chunk boundary
                current_chunk_text = " ".join(current_chunk_sentences)
                
                # Create boundary if:
                # 1. Similarity drops below threshold (semantic boundary)
                # 2. Chunk size would be exceeded by adding more sentences
                should_break = (
                    similarity < self.semantic_threshold or
                    len(current_chunk_text) >= self.chunk_size
                )
                
                if should_break or i == len(similarities) - 1:  # Last iteration
                    # Create chunk with overlap if needed
                    if chunks and self.chunk_overlap > 0:
                        overlap_text = self._get_semantic_overlap(chunks[-1]['content'], self.chunk_overlap)
                        if overlap_text:
                            current_chunk_text = overlap_text + " " + current_chunk_text
                    
                    chunks.append({
                        'content': current_chunk_text.strip(),
                        'chunk_index': chunk_index,
                        'metadata': {
                            **(metadata or {}),
                            'chunking_method': 'semantic',
                            'char_count': len(current_chunk_text),
                            'sentence_count': len(current_chunk_sentences),
                            'semantic_boundary': similarity < self.semantic_threshold if i < len(similarities) - 1 else False,
                            'similarity_score': similarity if i < len(similarities) - 1 else None
                        }
                    })
                    
                    # Start new chunk with potential overlap
                    if i < len(similarities) - 1:  # Not the last iteration
                        if self.chunk_overlap > 0:
                            # Keep some sentences for overlap
                            overlap_sentences = self._get_overlap_sentences(current_chunk_sentences, self.chunk_overlap)
                            current_chunk_sentences = overlap_sentences + [sentences[i + 1]]
                        else:
                            current_chunk_sentences = [sentences[i + 1]]
                        chunk_index += 1
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            return self._recursive_chunk_text(text, metadata)
    
    def _get_semantic_overlap(self, previous_chunk: str, overlap_chars: int) -> str:
        """Get semantic overlap from previous chunk"""
        if len(previous_chunk) <= overlap_chars:
            return previous_chunk
        
        # Try to break at sentence boundaries for better semantic overlap
        sentences = self._split_into_sentences(previous_chunk)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= overlap_chars:
                overlap_text = sentence + " " + overlap_text if overlap_text else sentence
            else:
                break
        
        return overlap_text.strip()
    
    def _agentic_chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Agentic chunking using LLM to determine optimal chunk boundaries
        """
        if not AGENTIC_CHUNKING_AVAILABLE:
            logger.warning("Agentic chunking not available, falling back to semantic chunking")
            fallback_chunks = self._semantic_chunk_text(text, metadata) if self.semantic_model else self._recursive_chunk_text(text, metadata)
            # Update metadata to reflect the actual method used
            for chunk in fallback_chunks:
                chunk['metadata']['chunking_method'] = 'semantic' if self.semantic_model else 'recursive'
                chunk['metadata']['original_request'] = 'agentic'
                chunk['metadata']['fallback_reason'] = 'agentic_chunking_not_available'
            return fallback_chunks
        
        try:
            # First, split into reasonable segments for LLM analysis
            segments = self._split_into_segments(text, max_segment_size=2000)
            
            chunks = []
            chunk_index = 0
            
            for segment in segments:
                # Use LLM to determine optimal chunk boundaries within this segment
                segment_chunks = self._llm_analyze_segment(segment, chunk_index)
                
                # Add overlap between segments if needed
                if chunks and segment_chunks and self.chunk_overlap > 0:
                    overlap_text = self._get_overlap_text(chunks[-1]['content'], self.chunk_overlap)
                    if overlap_text:
                        segment_chunks[0]['content'] = overlap_text + " " + segment_chunks[0]['content']
                        segment_chunks[0]['metadata']['has_overlap'] = True
                
                chunks.extend(segment_chunks)
                chunk_index += len(segment_chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in agentic chunking: {e}")
            fallback_chunks = self._semantic_chunk_text(text, metadata) if self.semantic_model else self._recursive_chunk_text(text, metadata)
            # Update metadata to reflect the actual method used
            for chunk in fallback_chunks:
                chunk['metadata']['chunking_method'] = 'semantic' if self.semantic_model else 'recursive'
                chunk['metadata']['original_request'] = 'agentic'
                chunk['metadata']['fallback_reason'] = str(e)
            return fallback_chunks
    
    def _split_into_segments(self, text: str, max_segment_size: int = 2000) -> List[str]:
        """Split text into segments for LLM analysis"""
        if len(text) <= max_segment_size:
            return [text]
        
        segments = []
        sentences = self._split_into_sentences(text)
        current_segment = ""
        
        for sentence in sentences:
            if len(current_segment + sentence) <= max_segment_size:
                current_segment += " " + sentence if current_segment else sentence
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence
        
        if current_segment:
            segments.append(current_segment.strip())
        
        return segments
    
    def _llm_analyze_segment(self, segment: str, start_index: int) -> List[Dict[str, Any]]:
        """Use LLM to analyze a segment and determine optimal chunk boundaries"""
        prompt = f"""
        Analyze the following text segment and determine the optimal places to split it into chunks of approximately {self.chunk_size} characters each.

        Consider:
        1. Semantic coherence - keep related ideas together
        2. Natural language boundaries - prefer to split at paragraph or sentence breaks
        3. Topic shifts - split when the topic changes significantly
        4. Logical flow - maintain logical progression within chunks

        Text segment:
        {segment}

        Respond with a JSON array of chunks, where each chunk is an object with:
        - "content": the text content of the chunk
        - "reasoning": brief explanation of why this boundary was chosen
        - "confidence": confidence score from 0.0 to 1.0

        Example format:
        [
            {{"content": "first chunk text...", "reasoning": "complete thought about topic A", "confidence": 0.9}},
            {{"content": "second chunk text...", "reasoning": "topic shift to B", "confidence": 0.8}}
        ]
        """
        
        try:
            if self.llm_provider == 'gemini' and self.gemini_api_key:
                model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
                response = model.generate_content(prompt)
                result = response.text
            else:
                # Fallback if no LLM provider available
                logger.warning(f"LLM provider {self.llm_provider} not available for agentic chunking, falling back")
                return self._sentence_chunk_text(segment, {'segment_fallback': True}, start_index)
            
            # Parse JSON response - handle markdown code blocks
            import json
            import re
            
            # Extract JSON from markdown code blocks if present
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', result, re.DOTALL)
            if json_match:
                json_text = json_match.group(1).strip()
            else:
                json_text = result.strip()
            
            # Try to parse the extracted JSON
            chunk_data = json.loads(json_text)
            
            # Convert to our chunk format
            chunks = []
            for i, chunk_info in enumerate(chunk_data):
                chunks.append({
                    'content': chunk_info['content'].strip(),
                    'chunk_index': start_index + i,
                    'metadata': {
                        'chunking_method': 'agentic',
                        'char_count': len(chunk_info['content']),
                        'llm_reasoning': chunk_info.get('reasoning', ''),
                        'llm_confidence': chunk_info.get('confidence', 0.5),
                        'llm_provider': self.llm_provider
                    }
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            # Fallback to sentence-based chunking for this segment
            fallback_chunks = self._sentence_chunk_text(segment, {'segment_fallback': True}, start_index)
            # Update metadata to reflect the actual method used
            for chunk in fallback_chunks:
                chunk['metadata']['chunking_method'] = 'sentence_based'
                chunk['metadata']['original_request'] = 'agentic'
                chunk['metadata']['fallback_reason'] = str(e)
            return fallback_chunks
    
    def _adaptive_intelligent_chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Enhanced adaptive chunking that intelligently selects the best method including agentic chunking
        """
        # Analyze text characteristics for intelligent method selection
        text_stats = self._analyze_text_characteristics(text)
        
        # Enhanced decision logic that includes agentic chunking
        primary_method = self._select_optimal_chunking_method(text_stats, text)
        
        logger.info(f"Adaptive chunking chose {primary_method} method based on text analysis: {text_stats}")
        
        # Apply chosen method with fallback chain
        try:
            if primary_method == 'agentic' and AGENTIC_CHUNKING_AVAILABLE:
                chunks = self._agentic_chunk_text(text, metadata)
            elif primary_method == 'semantic' and self.semantic_model:
                chunks = self._semantic_chunk_text(text, metadata)
            elif primary_method == 'recursive':
                chunks = self._recursive_chunk_text(text, metadata)
            else:
                # Fallback to sentence-based chunking
                chunks = self._sentence_chunk_text(text, metadata)
                primary_method = 'sentence'
        except Exception as e:
            logger.warning(f"Failed to use {primary_method} chunking, falling back: {e}")
            # Fallback chain: agentic -> semantic -> recursive -> sentence
            if primary_method == 'agentic' and self.semantic_model:
                chunks = self._semantic_chunk_text(text, metadata)
                primary_method = 'semantic_fallback'
            elif primary_method in ['agentic', 'semantic']:
                chunks = self._recursive_chunk_text(text, metadata)
                primary_method = 'recursive_fallback'
            else:
                chunks = self._sentence_chunk_text(text, metadata)
                primary_method = 'sentence_fallback'
        
        # Add adaptive metadata with decision reasoning
        for chunk in chunks:
            chunk['metadata'].update({
                'adaptive_method': primary_method,
                'text_analysis': text_stats,
                'chunking_method': primary_method,  # Store the actual method used, not "adaptive"
                'selected_strategy': primary_method,
                'adaptive_reasoning': 'auto_selected',
                'original_request': 'adaptive'  # Track that adaptive was requested
            })
        
        return chunks
    
    def _select_optimal_chunking_method(self, text_stats: Dict[str, Any], text: str) -> str:
        """
        Intelligent method selection logic for adaptive chunking
        """
        # Priority scoring for different methods based on content characteristics
        method_scores = {
            'agentic': 0,
            'semantic': 0, 
            'recursive': 0,
            'sentence': 0
        }
        
        # Agentic chunking criteria (highest quality but most expensive)
        if AGENTIC_CHUNKING_AVAILABLE and (self.llm_provider == 'gemini' and self.gemini_api_key):
            # Complex documents benefit from LLM intelligence
            if text_stats['is_complex']:
                method_scores['agentic'] += 50
            
            # Technical content with mixed structure
            if text_stats['technical_indicators'] > 5:
                method_scores['agentic'] += 30
            
            # Long documents with varied content
            if text_stats['length'] > 8000 and text_stats['paragraph_count'] > 10:
                method_scores['agentic'] += 40
            
            # Documents with both lists and narrative content
            if text_stats['has_lists'] and text_stats['paragraph_count'] > 5:
                method_scores['agentic'] += 25
            
            # Legal/academic documents (detected by formal structure)
            if (text_stats['avg_sentence_length'] > 150 and 
                text_stats['has_numbers'] and 
                text_stats['technical_indicators'] > 3):
                method_scores['agentic'] += 35
            
            # Organizational documents (camp handbooks, staff directories, contact info)
            if text_stats['organizational_indicators'] > 3:
                method_scores['agentic'] += 30
            
            # Documents with many organizational references benefit from intelligent boundary detection
            if (text_stats['organizational_indicators'] > 5 and 
                text_stats['paragraph_count'] > 3):
                method_scores['agentic'] += 25
        
        # Semantic chunking criteria (good for narrative content)
        if self.semantic_model:
            # Clear narrative structure
            if (text_stats['has_clear_structure'] and 
                not text_stats['has_lists'] and 
                text_stats['paragraph_count'] > 3):
                method_scores['semantic'] += 40
            
            # Medium complexity documents
            if 2000 < text_stats['length'] < 8000:
                method_scores['semantic'] += 30
            
            # Articles/blogs with good paragraph structure
            if (text_stats['paragraph_count'] > 5 and 
                50 < text_stats['avg_sentence_length'] < 150):
                method_scores['semantic'] += 35
            
            # Content without too many technical indicators
            if text_stats['technical_indicators'] < 10:
                method_scores['semantic'] += 20
            
            # Organizational documents with moderate complexity benefit from semantic chunking
            if (text_stats['organizational_indicators'] > 2 and 
                text_stats['organizational_indicators'] < 8 and
                text_stats['has_clear_structure']):
                method_scores['semantic'] += 25
        
        # Recursive chunking criteria (reliable fallback, good for structured content)
        # Always available as fallback
        method_scores['recursive'] = 25  # Base score
        
        # Structured documents (code, data, configuration)
        if text_stats['has_lists'] or text_stats['has_numbers']:
            method_scores['recursive'] += 30
        
        # Clear hierarchical structure
        if text_stats['has_clear_structure']:
            method_scores['recursive'] += 25
        
        # Technical documentation
        if text_stats['technical_indicators'] > 10:
            method_scores['recursive'] += 20
        
        # Very long documents where performance matters
        if text_stats['length'] > 15000:
            method_scores['recursive'] += 15
        
        # Sentence chunking criteria (simple and fast)
        method_scores['sentence'] = 10  # Base score
        
        # Short, simple documents
        if text_stats['length'] < 2000 and text_stats['paragraph_count'] < 3:
            method_scores['sentence'] += 30
        
        # Simple structure without complexity
        if (text_stats['avg_sentence_length'] < 100 and 
            not text_stats['has_lists'] and 
            text_stats['technical_indicators'] < 3):
            method_scores['sentence'] += 25
        
        # Select method with highest score
        best_method = max(method_scores.items(), key=lambda x: x[1])
        
        logger.info(f"Method scores: {method_scores}, selected: {best_method[0]} (score: {best_method[1]})")
        
        return best_method[0]
    
    def _analyze_text_characteristics(self, text: str) -> Dict[str, Any]:
        """
        Enhanced analysis of text characteristics for intelligent chunking method selection
        """
        lines = text.split('\n')
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Basic metrics
        line_count = len(lines)
        sentence_count = len(sentences)
        paragraph_count = len(paragraphs)
        word_count = len(text.split())
        char_count = len(text)
        
        # Advanced pattern detection for agentic chunking
        # Technical content indicators
        technical_patterns = [
            r'\b(?:function|class|method|algorithm|implementation|protocol|framework|architecture|specification)\b',
            r'\b(?:API|REST|HTTP|JSON|XML|SQL|database|server|client|endpoint)\b',
            r'\b(?:analysis|methodology|research|study|hypothesis|conclusion|results|findings)\b',
            r'\b(?:Section|Article|Chapter|Appendix|Figure|Table|Reference|Citation)\b',
            r'\b(?:shall|must|should|may|requirement|compliance|standard|regulation)\b',
            r'\b(?:whereas|therefore|furthermore|however|nevertheless|consequently)\b'
        ]
        
        technical_count = 0
        for pattern in technical_patterns:
            technical_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Legal/formal document indicators
        legal_patterns = [
            r'\b(?:pursuant|thereof|herein|whereas|heretofore|notwithstanding)\b',
            r'\b(?:contract|agreement|clause|provision|liability|indemnification)\b',
            r'\b(?:party|parties|signatory|witness|notary|jurisdiction)\b'
        ]
        
        legal_count = 0
        for pattern in legal_patterns:
            legal_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Academic/research indicators  
        academic_patterns = [
            r'\b(?:abstract|introduction|methodology|literature|review|discussion)\b',
            r'\b(?:hypothesis|variables|correlation|significance|p-value|statistical)\b',
            r'\b(?:et al\.|ibid\.|op\. cit\.|cf\.|i\.e\.|e\.g\.)\b'
        ]
        
        academic_count = 0
        for pattern in academic_patterns:
            academic_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Organizational/personnel indicators (for better handling of staff/contact info)
        organizational_patterns = [
            r'\b(?:director|manager|administrator|coordinator|supervisor|leader|head|chief)\b',
            r'\b(?:staff|employee|personnel|team|contact|phone|email|address)\b',
            r'\b(?:office|department|division|center|facility|location|headquarters)\b',
            r'\b(?:founded|established|operated|managed|directed|supervised)\b',
            r'\b(?:mr\.|mrs\.|ms\.|dr\.|prof\.|president|vice|assistant|associate)\b',
            r'\b(?:camp|school|organization|company|institution|foundation|association)\b'
        ]
        
        organizational_count = 0
        for pattern in organizational_patterns:
            organizational_count += len(re.findall(pattern, text, re.IGNORECASE))
        
        # Structure analysis
        has_headings = bool(re.search(r'^#{1,6}\s+.+$', text, re.MULTILINE))
        has_bullet_points = bool(re.search(r'^\s*[-*+•]\s+.+$', text, re.MULTILINE))
        has_numbered_lists = bool(re.search(r'^\s*\d+[\.\)]\s+.+$', text, re.MULTILINE))
        has_tables = bool(re.search(r'\|.*\|', text))
        has_code_blocks = bool(re.search(r'```|`[^`]+`', text))
        
        # Complexity indicators for agentic chunking
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(sentence_count, 1)
        avg_paragraph_length = char_count / max(paragraph_count, 1)
        
        # Mixed content detection (suggests need for intelligent chunking)
        content_types = sum([
            has_headings,
            has_bullet_points, 
            has_numbered_lists,
            has_tables,
            has_code_blocks,
            bool(re.search(r'\d+', text)),  # has numbers
            bool(re.search(r'[A-Z]{2,}', text))  # has acronyms
        ])
        
        # Complex document detection for agentic chunking
        is_complex = (
            (technical_count + legal_count + academic_count + organizational_count > 10) or  # High technical/formal content
            (content_types >= 4) or  # Mixed content types
            (avg_sentence_length > 200) or  # Very long sentences
            (paragraph_count > 15 and char_count > 10000) or  # Long structured document
            (legal_count > 5) or  # Legal document
            (academic_count > 8) or  # Academic paper
            (organizational_count > 5)  # Organizational documents with many staff/contact references
        )
        
        return {
            'length': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'paragraph_count': paragraph_count,
            'line_count': line_count,
            'avg_sentence_length': avg_sentence_length,
            'avg_paragraph_length': avg_paragraph_length,
            
            # Structure indicators
            'has_headings': has_headings,
            'has_lists': has_bullet_points or has_numbered_lists,
            'has_bullet_points': has_bullet_points,
            'has_numbered_lists': has_numbered_lists,
            'has_tables': has_tables,
            'has_code_blocks': has_code_blocks,
            'has_numbers': bool(re.search(r'\d+', text)),
            'has_clear_structure': has_headings or (paragraph_count > 2 and avg_paragraph_length > 100),
            
            # Content type indicators  
            'technical_indicators': technical_count,
            'legal_indicators': legal_count,
            'academic_indicators': academic_count,
            'organizational_indicators': organizational_count,
            'content_type_diversity': content_types,
            
            # Complexity assessment for method selection
            'is_complex': is_complex,
            'complexity_score': technical_count + legal_count + academic_count + organizational_count + content_types,
            
            # Agentic chunking triggers
            'needs_intelligent_chunking': (
                is_complex or 
                (content_types >= 3 and char_count > 5000) or
                legal_count > 3 or
                academic_count > 5 or
                organizational_count > 3  # Documents with organizational information benefit from intelligent chunking
            )
        }
    
    def _character_level_split(self, text: str, metadata: Dict[str, Any], start_index: int = 0) -> List[Dict[str, Any]]:
        """Character-level splitting as last resort"""
        chunks = []
        chunk_index = start_index
        
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # Add overlap from previous chunk
            if start > 0 and self.chunk_overlap > 0:
                overlap_start = max(0, start - self.chunk_overlap)
                chunk_text = text[overlap_start:end]
            else:
                chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    'content': chunk_text.strip(),
                    'chunk_index': chunk_index,
                    'metadata': {
                        **(metadata or {}),
                        'chunking_method': 'character_level',
                        'char_count': len(chunk_text.strip()),
                        'separator_used': 'character'
                    }
                })
                chunk_index += 1
            
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _sentence_chunk_text(self, text: str, metadata: Dict[str, Any] = None, start_index: int = 0) -> List[Dict[str, Any]]:
        """
        Original sentence-based chunking method (kept for backward compatibility)
        """
        chunks = []
        
        # Split text into sentences for better chunk boundaries
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_chunk_sentences = []
        chunk_index = start_index
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunks.append({
                    'content': current_chunk.strip(),
                    'chunk_index': chunk_index,
                    'metadata': {
                        **(metadata or {}),
                        'chunking_method': 'sentence_based',
                        'sentence_count': len(current_chunk_sentences),
                        'char_count': len(current_chunk)
                    }
                })
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk_sentences, self.chunk_overlap
                )
                current_chunk = " ".join(overlap_sentences)
                current_chunk_sentences = overlap_sentences.copy()
                chunk_index += 1
            
            # Add sentence to current chunk
            current_chunk += " " + sentence if current_chunk else sentence
            current_chunk_sentences.append(sentence)
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'chunk_index': chunk_index,
                'metadata': {
                    **(metadata or {}),
                    'chunking_method': 'sentence_based',
                    'sentence_count': len(current_chunk_sentences),
                    'char_count': len(current_chunk)
                }
            })
        
        return chunks
    
    def _get_overlap_text(self, text: str, overlap_chars: int) -> str:
        """
        Get overlap text from the end of current chunk
        """
        if len(text) <= overlap_chars:
            return text
        
        # Try to find a good breaking point (space, punctuation)
        overlap_text = text[-overlap_chars:]
        
        # Look for a space to break on
        space_index = overlap_text.find(' ')
        if space_index > 0:
            return overlap_text[space_index + 1:]
        
        return overlap_text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex patterns
        """
        # Pattern to split on sentence endings, but preserve them
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_chars: int) -> List[str]:
        """
        Get sentences for overlap based on character count
        """
        if not sentences:
            return []
        
        overlap_sentences = []
        char_count = 0
        
        # Work backwards from the end of sentences
        for sentence in reversed(sentences):
            if char_count + len(sentence) <= overlap_chars:
                overlap_sentences.insert(0, sentence)
                char_count += len(sentence)
            else:
                break
        
        return overlap_sentences
    
    def process_pdf_content(self, content: str, s3_key: str) -> List[Dict[str, Any]]:
        """
        Process PDF content and create chunks with specific metadata
        """
        metadata = {
            'source_type': 'pdf',
            's3_key': s3_key,
            'content_type': 'application/pdf'
        }
        
        # Clean PDF content
        cleaned_content = self._clean_pdf_text(content)
        
        # Use the configured chunking method
        return self.chunk_text(cleaned_content, metadata)
    
    def process_text_content(self, content: str, s3_key: str, content_type: str = 'text/plain') -> List[Dict[str, Any]]:
        """
        Process text content and create chunks
        """
        metadata = {
            'source_type': 'text',
            's3_key': s3_key,
            'content_type': content_type
        }
        
        # Use the configured chunking method
        return self.chunk_text(content, metadata)
    
    def process_json_content(self, content: str, s3_key: str) -> List[Dict[str, Any]]:
        """
        Process JSON content and create chunks
        """
        metadata = {
            'source_type': 'json',
            's3_key': s3_key,
            'content_type': 'application/json'
        }
        
        # Use the configured chunking method
        return self.chunk_text(content, metadata)
    
    def process_image_content(self, content: str, s3_key: str) -> List[Dict[str, Any]]:
        """
        Process image OCR content and create chunks
        """
        metadata = {
            'source_type': 'image',
            's3_key': s3_key,
            'content_type': 'image/ocr',
            'extraction_method': 'tesseract_ocr'
        }
        
        # Use the configured chunking method
        return self.chunk_text(content, metadata)
    
    def process_excel_content(self, content: str, s3_key: str, file_type: str = 'excel') -> List[Dict[str, Any]]:
        """
        Process Excel content and create chunks
        """
        metadata = {
            'source_type': 'excel',
            's3_key': s3_key,
            'content_type': f'application/{file_type}',
            'extraction_method': 'xlrd' if file_type == 'xls' else 'openpyxl'
        }
        
        # Use the configured chunking method
        return self.chunk_text(content, metadata)
    
    def process_docx_content(self, content: str, s3_key: str) -> List[Dict[str, Any]]:
        """
        Process Word document content and create chunks
        """
        metadata = {
            'source_type': 'docx',
            's3_key': s3_key,
            'content_type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'extraction_method': 'python-docx'
        }
        
        # Use the configured chunking method
        return self.chunk_text(content, metadata)
    
    def _clean_pdf_text(self, text: str) -> str:
        """
        Clean PDF extracted text by removing extra whitespace and fixing common issues
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page breaks and form feeds
        text = re.sub(r'[\f\r]+', ' ', text)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'-\s+', '', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def get_optimal_chunking_strategy(self, s3_key: str, content: str) -> str:
        """
        Determine the optimal chunking strategy based on document type and content
        """
        file_type = self.get_file_type(s3_key)
        content_length = len(content)
        
        # For structured documents, prefer recursive chunking
        if file_type in ['json', 'csv']:
            return 'recursive'
        
        # For long documents, recursive chunking works better
        if content_length > 10000:
            return 'recursive'
        
        # Check if document has clear structure (many paragraphs)
        paragraph_count = content.count('\n\n')
        if paragraph_count > 5:
            return 'recursive'
        
        # For short or simple documents, sentence-based is fine
        return 'sentence'
    
    def chunk_text_adaptive(self, text: str, s3_key: str = None, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Adaptive chunking that chooses the best strategy based on content
        """
        if not text or not text.strip():
            return []
        
        # Determine optimal strategy
        if s3_key:
            strategy = self.get_optimal_chunking_strategy(s3_key, text)
        else:
            strategy = 'recursive' if self.use_recursive_chunking else 'sentence'
        
        # Temporarily override chunking method
        original_setting = self.use_recursive_chunking
        self.use_recursive_chunking = (strategy == 'recursive')
        
        try:
            chunks = self.chunk_text(text, metadata)
            # Add strategy info to metadata
            for chunk in chunks:
                chunk['metadata']['chosen_strategy'] = strategy
                # Update chunking_method to reflect the actual method used, not "adaptive"
                if 'chunking_method' not in chunk['metadata'] or chunk['metadata']['chunking_method'] == 'adaptive':
                    chunk['metadata']['chunking_method'] = strategy
                chunk['metadata']['adaptive_reasoning'] = 'auto_selected'
                chunk['metadata']['original_request'] = 'adaptive'  # Track that adaptive was requested
            return chunks
        finally:
            self.use_recursive_chunking = original_setting
    
    def get_file_type(self, s3_key: str) -> str:
        """
        Determine file type from S3 key extension
        """
        if not s3_key:
            return 'text'
        
        # Extract file extension
        file_ext = s3_key.lower().split('.')[-1] if '.' in s3_key else ''
        
        # Map extensions to file types
        extension_map = {
            'pdf': 'pdf',
            'txt': 'text',
            'md': 'text',
            'markdown': 'text',
            'json': 'json',
            'csv': 'csv',
            'tsv': 'csv',
            'log': 'log',
            'py': 'code',
            'js': 'code',
            'html': 'markup',
            'xml': 'markup',
            'yml': 'config',
            'yaml': 'config',
            # New supported file types
            'jpg': 'image',
            'jpeg': 'image',
            'png': 'image',
            'tiff': 'image',
            'bmp': 'image',
            'gif': 'image',
            'xls': 'excel',
            'xlsx': 'excel',
            'doc': 'docx',
            'docx': 'docx'
        }
        
        return extension_map.get(file_ext, 'text')
    
    def create_search_context(self, chunks: List[Dict[str, Any]], max_length: int = None) -> str:
        """
        Create search context from multiple chunks, ensuring we don't exceed max length
        """
        if not chunks:
            return ""
        
        max_length = max_length or self.max_context_length
        context_parts = []
        total_length = 0
        
        for chunk in chunks:
            content = chunk['content']
            s3_key = chunk.get('s3_key', 'unknown')
            
            # Format chunk with source information
            chunk_text = f"[Source: {s3_key}]\n{content}\n"
            
            # Check if adding this chunk would exceed max length
            if total_length + len(chunk_text) > max_length:
                break
            
            context_parts.append(chunk_text)
            total_length += len(chunk_text)
        
        return "\n".join(context_parts)
