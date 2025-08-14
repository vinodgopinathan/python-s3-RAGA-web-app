#!/usr/bin/env python3
"""
Advanced Chunking Test Suite
Tests all chunking methods: recursive, semantic, agentic, and adaptive
"""

import os
import sys
import argparse
import time
from typing import Dict, Any, List
import json

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.document_processor import DocumentProcessor

def create_test_documents() -> Dict[str, str]:
    """Create test documents of different types"""
    return {
        "technical_doc": """
        Machine Learning Model Architecture

        Our system employs a multi-layered neural network architecture designed for natural language processing tasks. The architecture consists of several key components that work together to achieve state-of-the-art performance.

        Embedding Layer
        The embedding layer converts input tokens into dense vector representations. We use a 768-dimensional embedding space with vocabulary size of 50,000 tokens. The embeddings are initialized using GloVe pre-trained vectors and fine-tuned during training.

        Attention Mechanism
        The attention mechanism allows the model to focus on relevant parts of the input sequence. We implement multi-head attention with 12 attention heads, each with a dimension of 64. This enables the model to capture different types of relationships simultaneously.

        Feed-Forward Networks
        Each transformer block contains a feed-forward network with hidden dimension 3072. We use GELU activation function and apply dropout with rate 0.1 for regularization.

        Training Process
        The model is trained using Adam optimizer with learning rate 2e-5. We employ gradient clipping with maximum norm 1.0 to prevent exploding gradients. The training process includes warmup for the first 10% of steps.

        Evaluation Metrics
        We evaluate the model using BLEU score, ROUGE metrics, and human evaluation. The model achieves BLEU-4 score of 28.5 on the test set, which represents a 15% improvement over the baseline.
        """,
        
        "narrative_text": """
        The old lighthouse stood majestically on the rocky cliff, its weathered walls telling stories of countless storms weathered and ships guided safely to shore. Sarah had visited this place many times as a child, but today felt different.

        She climbed the winding stairs, each step echoing in the hollow tower. The keeper's quarters had been abandoned for decades, yet somehow the place still felt alive with memories. Dust particles danced in the afternoon sunlight streaming through the salt-stained windows.

        At the top, she found what she had been looking for. Hidden behind the massive lens apparatus was a small wooden box, exactly where her grandfather had told her it would be. Inside were letters, tied with a faded blue ribbon.

        The first letter was dated 1943. It was from her grandmother to her grandfather, written during the war when he was stationed at sea. The words spoke of love, hope, and the beacon of light that would guide him home.

        As Sarah read through the letters, she began to understand the depth of their love story. Each letter revealed more about their struggles, their dreams, and their unwavering commitment to each other despite the distance and uncertainty of war.

        The lighthouse had been more than just a navigational aid; it had been a symbol of hope and homecoming for two young people in love.
        """,
        
        "structured_data": """
        API Documentation: User Management System

        1. Authentication Endpoints
           1.1 POST /auth/login
               - Parameters: email, password
               - Returns: JWT token, user profile
               - Status Codes: 200 (success), 401 (invalid credentials), 400 (bad request)
           
           1.2 POST /auth/register
               - Parameters: email, password, firstName, lastName
               - Returns: User ID, confirmation message
               - Status Codes: 201 (created), 409 (email exists), 400 (validation error)

        2. User Profile Endpoints
           2.1 GET /users/profile
               - Headers: Authorization (Bearer token)
               - Returns: Complete user profile
               - Status Codes: 200 (success), 401 (unauthorized)
           
           2.2 PUT /users/profile
               - Headers: Authorization (Bearer token)
               - Parameters: firstName, lastName, phone, address
               - Returns: Updated profile
               - Status Codes: 200 (updated), 401 (unauthorized), 400 (validation error)

        3. Data Models
           3.1 User Model
               - id: UUID (primary key)
               - email: String (unique, required)
               - password: String (hashed, required)
               - firstName: String (required)
               - lastName: String (required)
               - phone: String (optional)
               - address: Object (optional)
               - createdAt: DateTime
               - updatedAt: DateTime

        4. Error Handling
           All endpoints return standardized error responses with status codes and descriptive messages.
        """
    }

def test_chunking_method(processor: DocumentProcessor, text: str, method: str, doc_name: str) -> Dict[str, Any]:
    """Test a specific chunking method"""
    print(f"\n{'='*60}")
    print(f"Testing {method.upper()} chunking on {doc_name}")
    print(f"{'='*60}")
    
    # Set chunking method
    processor.chunking_method = method
    
    # Start timing
    start_time = time.time()
    
    # Perform chunking
    metadata = {'test_document': doc_name, 'test_method': method}
    chunks = processor.chunk_text(text, metadata)
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Calculate statistics
    chunk_sizes = [len(chunk['content']) for chunk in chunks]
    
    results = {
        'method': method,
        'document': doc_name,
        'chunk_count': len(chunks),
        'total_chars': len(text),
        'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
        'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
        'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
        'processing_time': processing_time,
        'chunks_preview': []
    }
    
    # Print summary
    print(f"Document length: {len(text)} characters")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Average chunk size: {results['avg_chunk_size']:.1f} characters")
    print(f"Size range: {results['min_chunk_size']} - {results['max_chunk_size']}")
    print(f"Processing time: {processing_time:.3f} seconds")
    
    # Show chunk previews
    print(f"\nChunk previews:")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        preview = chunk['content'][:100] + "..." if len(chunk['content']) > 100 else chunk['content']
        print(f"  Chunk {i+1}: {preview}")
        
        # Store detailed info for first few chunks
        if i < 3:
            results['chunks_preview'].append({
                'index': chunk['chunk_index'],
                'content_preview': preview,
                'full_length': len(chunk['content']),
                'metadata': chunk.get('metadata', {})
            })
    
    if len(chunks) > 3:
        print(f"  ... and {len(chunks) - 3} more chunks")
    
    return results

def compare_methods(results: List[Dict[str, Any]]) -> None:
    """Compare results across different methods"""
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    # Group by document
    by_document = {}
    for result in results:
        doc = result['document']
        if doc not in by_document:
            by_document[doc] = []
        by_document[doc].append(result)
    
    for doc_name, doc_results in by_document.items():
        print(f"\n{doc_name.upper()}:")
        print(f"{'Method':<12} {'Chunks':<8} {'Avg Size':<10} {'Time':<8} {'Notes'}")
        print("-" * 60)
        
        for result in doc_results:
            notes = ""
            if result['method'] == 'semantic' and any('semantic_boundary' in chunk.get('metadata', {}) for chunk in result.get('chunks_preview', [])):
                notes = "Semantic boundaries detected"
            elif result['method'] == 'agentic':
                notes = "LLM-guided boundaries"
            elif result['method'] == 'adaptive':
                notes = "Auto-selected best method"
            
            print(f"{result['method']:<12} {result['chunk_count']:<8} {result['avg_chunk_size']:<10.1f} {result['processing_time']:<8.3f} {notes}")

def run_comprehensive_test():
    """Run comprehensive chunking tests"""
    print("🧠 Advanced Chunking Test Suite")
    print("Testing: Recursive, Semantic, Agentic, and Adaptive chunking methods")
    
    # Create document processor
    processor = DocumentProcessor()
    
    # Get test documents
    documents = create_test_documents()
    
    # Methods to test
    methods = ['recursive', 'semantic', 'agentic', 'adaptive']
    
    # Store all results
    all_results = []
    
    # Test each method on each document
    for doc_name, text in documents.items():
        for method in methods:
            try:
                result = test_chunking_method(processor, text, method, doc_name)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error testing {method} on {doc_name}: {e}")
                continue
    
    # Compare results
    compare_methods(all_results)
    
    # Save detailed results
    output_file = "chunking_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {output_file}")
    
    return all_results

def test_file_chunking(file_path: str, methods: List[str] = None):
    """Test chunking on a specific file"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    methods = methods or ['recursive', 'semantic', 'agentic', 'adaptive']
    
    print(f"📄 Testing chunking on file: {file_path}")
    
    # Read file content
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Create processor
    processor = DocumentProcessor()
    
    # Test each method
    results = []
    for method in methods:
        try:
            result = test_chunking_method(processor, content, method, os.path.basename(file_path))
            results.append(result)
        except Exception as e:
            print(f"❌ Error testing {method}: {e}")
    
    # Compare results
    if results:
        compare_methods(results)

def main():
    parser = argparse.ArgumentParser(description='Test advanced chunking methods')
    parser.add_argument('--file', type=str, help='Test chunking on a specific file')
    parser.add_argument('--methods', nargs='+', choices=['recursive', 'semantic', 'agentic', 'adaptive'], 
                       default=['recursive', 'semantic', 'agentic', 'adaptive'],
                       help='Chunking methods to test')
    parser.add_argument('--comprehensive', action='store_true', 
                       help='Run comprehensive test on built-in documents')
    
    args = parser.parse_args()
    
    if args.file:
        test_file_chunking(args.file, args.methods)
    elif args.comprehensive:
        run_comprehensive_test()
    else:
        # Default: run comprehensive test
        run_comprehensive_test()

if __name__ == "__main__":
    main()
