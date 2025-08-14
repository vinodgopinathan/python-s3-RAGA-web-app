#!/usr/bin/env python3
"""
Test script to demonstrate and compare different chunking strategies
"""

import os
import sys
import argparse
from pathlib import Path

# Add the backend source directory to Python path
backend_path = Path(__file__).parent / "src"
sys.path.insert(0, str(backend_path))

from src.utils.document_processor import DocumentProcessor

# Sample texts for testing
SAMPLE_TEXTS = {
    "structured": """
Introduction
============

This document covers machine learning fundamentals.

Chapter 1: Supervised Learning
==============================

Supervised learning is a type of machine learning where the algorithm learns from labeled training data.

1.1 Linear Regression
--------------------
Linear regression is used for predicting continuous values.

1.2 Classification
------------------
Classification is used for predicting discrete categories.

Chapter 2: Unsupervised Learning
================================

Unsupervised learning finds patterns in data without labeled examples.

2.1 Clustering
--------------
Clustering groups similar data points together.

Conclusion
==========

Machine learning has many applications in various domains.
""",
    
    "narrative": """
The sun was setting behind the mountains as Sarah walked down the winding path. She had been hiking for hours, but the peaceful sounds of nature kept her motivated. Birds chirped in the distance, and a gentle breeze rustled through the leaves. 

As she rounded the corner, she saw a beautiful lake reflecting the orange and pink hues of the sky. This was exactly what she had been looking for - a moment of tranquility away from the busy city life. She sat down on a large rock and pulled out her journal.

Writing had always been her way of processing the day's events. Today had been particularly challenging at work, with back-to-back meetings and urgent deadlines. But here, surrounded by nature's beauty, those stresses seemed to melt away.

She began to write about her thoughts and feelings, describing the serenity of the moment and her gratitude for being able to escape to this peaceful place.
""",
    
    "technical": """
Data preprocessing is a crucial step in machine learning pipelines. It involves cleaning, transforming, and organizing raw data. Key steps include: handling missing values, removing outliers, normalizing data, encoding categorical variables, and feature selection. Missing values can be handled through deletion, imputation, or prediction. Outliers should be identified using statistical methods or visualization techniques. Normalization ensures all features have similar scales. Categorical encoding converts text labels to numerical values. Feature selection reduces dimensionality and improves model performance.
"""
}

def read_file_content(file_path: str) -> str:
    """Read content from a file and return as string"""
    try:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Handle different file types
        if file_path.suffix.lower() == '.pdf':
            # For PDF files, you'd need PyPDF2 (already in requirements)
            try:
                import PyPDF2
                import io
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    content = ""
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
                return content
            except ImportError:
                raise ImportError("PyPDF2 is required to read PDF files")
        
        elif file_path.suffix.lower() in ['.txt', '.md', '.markdown', '.json', '.csv', '.log']:
            # For text-based files
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Could not decode file {file_path}")
        
        else:
            # Try as text file for unknown extensions
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
                
    except Exception as e:
        print(f"❌ Error reading file {file_path}: {e}")
        return None

def test_file_chunking(file_path: str, processor: DocumentProcessor):
    """Test chunking strategies on a specific file"""
    print(f"\n🔍 Testing file: {file_path}")
    print("=" * 80)
    
    content = read_file_content(file_path)
    if content is None:
        return None
    
    file_size = len(content)
    file_type = Path(file_path).suffix.lower()
    
    print(f"📁 File type: {file_type}")
    print(f"📏 File size: {file_size:,} characters")
    print(f"📄 Content preview: {content[:200]}...")
    print("-" * 80)
    
    # Test different strategies
    strategies = [
        ('recursive', True),
        ('sentence', False),
        ('adaptive', None)
    ]
    
    results = {}
    
    for strategy_name, use_recursive in strategies:
        print(f"\n🧪 Testing {strategy_name.upper()} chunking:")
        
        if strategy_name == 'adaptive':
            chunks = processor.chunk_text_adaptive(content, file_path, {'test_file': True})
        else:
            original_setting = processor.use_recursive_chunking
            processor.use_recursive_chunking = use_recursive
            
            try:
                chunks = processor.chunk_text(content, {'test_file': True})
            finally:
                processor.use_recursive_chunking = original_setting
        
        # Analyze results
        chunk_count = len(chunks)
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        results[strategy_name] = {
            'chunk_count': chunk_count,
            'chunk_sizes': chunk_sizes,
            'avg_size': avg_size,
            'chunks': chunks
        }
        
        print(f"   📊 Chunks created: {chunk_count}")
        print(f"   📏 Average size: {avg_size:.1f} characters")
        print(f"   📈 Size range: {min(chunk_sizes) if chunk_sizes else 0} - {max(chunk_sizes) if chunk_sizes else 0}")
        
        # Show chunking details
        if chunks:
            methods = set(chunk['metadata'].get('chunking_method', 'unknown') for chunk in chunks)
            separators = set(chunk['metadata'].get('separator_used', 'none') for chunk in chunks)
            print(f"   🔧 Methods used: {', '.join(methods)}")
            if 'none' not in separators:
                print(f"   ✂️  Separators used: {', '.join(repr(s) for s in separators if s != 'none')}")
    
    # Compare strategies
    print(f"\n📊 STRATEGY COMPARISON:")
    print("-" * 40)
    for strategy, data in results.items():
        consistency = calculate_consistency(data['chunk_sizes'])
        print(f"{strategy.capitalize():>12}: {data['chunk_count']:>3} chunks, "
              f"avg {data['avg_size']:>6.1f} chars, consistency: {consistency}")
    
    return results

def calculate_consistency(sizes):
    """Calculate consistency score for chunk sizes"""
    if len(sizes) <= 1:
        return "N/A"
    
    avg = sum(sizes) / len(sizes)
    variance = sum((size - avg) ** 2 for size in sizes) / len(sizes)
    std_dev = variance ** 0.5
    cv = std_dev / avg if avg > 0 else 0
    
    if cv < 0.2:
        return "Excellent"
    elif cv < 0.4:
        return "Good"
    elif cv < 0.6:
        return "Fair"
    else:
        return "Poor"
    """Test different chunking strategies on sample texts"""
    processor = DocumentProcessor()
    
    print("=== CHUNKING STRATEGY COMPARISON ===\n")
    
    for text_type, text in SAMPLE_TEXTS.items():
        print(f"🔍 Testing {text_type.upper()} text ({len(text)} characters):")
        print("-" * 60)
        
        # Test with recursive chunking
        processor.use_recursive_chunking = True
        recursive_chunks = processor.chunk_text(text, {'test_type': text_type})
        
        # Test with sentence-based chunking
        processor.use_recursive_chunking = False
        sentence_chunks = processor.chunk_text(text, {'test_type': text_type})
        
        # Test adaptive chunking
        adaptive_chunks = processor.chunk_text_adaptive(text, f"test_{text_type}.txt", {'test_type': text_type})
        
        print(f"📊 Results:")
        print(f"   Recursive chunking:  {len(recursive_chunks)} chunks")
        print(f"   Sentence chunking:   {len(sentence_chunks)} chunks")
        print(f"   Adaptive chunking:   {len(adaptive_chunks)} chunks")
        
        print(f"\n📝 Sample chunks (Recursive):")
        for i, chunk in enumerate(recursive_chunks[:2]):
            method = chunk['metadata'].get('chunking_method', 'unknown')
            separator = chunk['metadata'].get('separator_used', 'none')
            print(f"   Chunk {i+1} ({method}, sep={separator}):")
            print(f"   \"{chunk['content'][:100]}{'...' if len(chunk['content']) > 100 else ''}\"")
        
        print(f"\n📝 Sample chunks (Adaptive):")
        for i, chunk in enumerate(adaptive_chunks[:2]):
            method = chunk['metadata'].get('chunking_method', 'unknown')
            strategy = chunk['metadata'].get('chosen_strategy', 'unknown')
            print(f"   Chunk {i+1} ({method}, strategy={strategy}):")
            print(f"   \"{chunk['content'][:100]}{'...' if len(chunk['content']) > 100 else ''}\"")
        
        print("\n" + "="*80 + "\n")

def test_separator_effectiveness():
    """Test how different separators work with recursive chunking"""
    processor = DocumentProcessor()
    processor.use_recursive_chunking = True
    
    print("=== SEPARATOR EFFECTIVENESS TEST ===\n")
    
    test_text = SAMPLE_TEXTS["structured"]
    chunks = processor.chunk_text(test_text)
    
    separator_usage = {}
    for chunk in chunks:
        sep = chunk['metadata'].get('separator_used', 'none')
        separator_usage[sep] = separator_usage.get(sep, 0) + 1
    
    print("📊 Separator usage in recursive chunking:")
    for sep, count in separator_usage.items():
        print(f"   {sep}: {count} chunks")
    
    print(f"\n📋 Total chunks created: {len(chunks)}")
    avg_size = sum(len(chunk['content']) for chunk in chunks) / len(chunks)
    print(f"📏 Average chunk size: {avg_size:.1f} characters")

def compare_chunk_quality():
    """Compare chunk quality metrics between strategies"""
    processor = DocumentProcessor()
    
    print("=== CHUNK QUALITY COMPARISON ===\n")
    
    text = SAMPLE_TEXTS["structured"]
    
    # Test both strategies
    strategies = [
        ("Recursive", True),
        ("Sentence-based", False)
    ]
    
    for strategy_name, use_recursive in strategies:
        processor.use_recursive_chunking = use_recursive
        chunks = processor.chunk_text(text)
        
        # Calculate metrics
        sizes = [len(chunk['content']) for chunk in chunks]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        size_variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        
        print(f"📈 {strategy_name} chunking metrics:")
        print(f"   Chunks created: {len(chunks)}")
        print(f"   Average size: {avg_size:.1f} chars")
        print(f"   Size range: {min_size} - {max_size} chars")
        print(f"   Size variance: {size_variance:.1f}")
        print(f"   Size consistency: {'Good' if size_variance < 10000 else 'Poor'}")
        print()

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ.setdefault('CHUNK_SIZE', '500')  # Smaller for testing
    os.environ.setdefault('CHUNK_OVERLAP', '50')
    os.environ.setdefault('USE_RECURSIVE_CHUNKING', 'true')
    
    print("🧪 Document Processor Chunking Strategy Tests")
    print("=" * 60)
    print()
    
    test_chunking_strategies()
    test_separator_effectiveness()
    compare_chunk_quality()
    
def test_chunking_strategies():
    """Test different chunking strategies on sample texts"""
    processor = DocumentProcessor()
    
    print("=== CHUNKING STRATEGY COMPARISON (Sample Texts) ===\n")
    
    for text_type, text in SAMPLE_TEXTS.items():
        print(f"🔍 Testing {text_type.upper()} text ({len(text)} characters):")
        print("-" * 60)
        
        # Test with recursive chunking
        processor.use_recursive_chunking = True
        recursive_chunks = processor.chunk_text(text, {'test_type': text_type})
        
        # Test with sentence-based chunking
        processor.use_recursive_chunking = False
        sentence_chunks = processor.chunk_text(text, {'test_type': text_type})
        
        # Test adaptive chunking
        adaptive_chunks = processor.chunk_text_adaptive(text, f"test_{text_type}.txt", {'test_type': text_type})
        
        print(f"📊 Results:")
        print(f"   Recursive chunking:  {len(recursive_chunks)} chunks")
        print(f"   Sentence chunking:   {len(sentence_chunks)} chunks")
        print(f"   Adaptive chunking:   {len(adaptive_chunks)} chunks")
        
        # Show separator usage in recursive chunking
        if recursive_chunks:
            separators = {}
            for chunk in recursive_chunks:
                sep = chunk['metadata'].get('separator_used', 'none')
                separators[sep] = separators.get(sep, 0) + 1
            
            print(f"📊 Separator usage in recursive chunking:")
            for sep, count in separators.items():
                print(f"   {sep}: {count} chunks")
        
        print("\n" + "="*80 + "\n")

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Test chunking strategies on files or sample texts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with sample texts only
  python test_chunking.py
  
  # Test specific file
  python test_chunking.py --file /path/to/document.txt
  
  # Test multiple files
  python test_chunking.py --file doc1.pdf --file doc2.txt
  
  # Test with custom chunk size
  python test_chunking.py --file document.pdf --chunk-size 500 --overlap 100
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        action='append',
        dest='files',
        help='Path to file(s) to test chunking on (can be used multiple times)'
    )
    
    parser.add_argument(
        '--chunk-size', '-s',
        type=int,
        help='Chunk size for testing (default: 1000)'
    )
    
    parser.add_argument(
        '--overlap', '-o',
        type=int,
        help='Chunk overlap for testing (default: 200)'
    )
    
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Disable recursive chunking (use sentence-based only)'
    )
    
    parser.add_argument(
        '--strategy',
        choices=['recursive', 'sentence', 'adaptive', 'all'],
        default='all',
        help='Test specific strategy only (default: all)'
    )
    
    args = parser.parse_args()
    
    # Set environment variables for testing
    if args.chunk_size:
        os.environ['CHUNK_SIZE'] = str(args.chunk_size)
    else:
        os.environ.setdefault('CHUNK_SIZE', '1000')
    
    if args.overlap:
        os.environ['CHUNK_OVERLAP'] = str(args.overlap)
    else:
        os.environ.setdefault('CHUNK_OVERLAP', '200')
    
    if args.no_recursive:
        os.environ['USE_RECURSIVE_CHUNKING'] = 'false'
    else:
        os.environ.setdefault('USE_RECURSIVE_CHUNKING', 'true')
    
    print("🧪 Document Processor Chunking Strategy Tests")
    print("=" * 60)
    print(f"📐 Chunk size: {os.environ.get('CHUNK_SIZE')}")
    print(f"🔄 Overlap: {os.environ.get('CHUNK_OVERLAP')}")
    print(f"🔀 Recursive chunking: {os.environ.get('USE_RECURSIVE_CHUNKING')}")
    print()
    
    processor = DocumentProcessor()
    
    # Test files if provided
    if args.files:
        print("📁 TESTING YOUR FILES")
        print("=" * 40)
        
        all_results = {}
        for file_path in args.files:
            result = test_file_chunking(file_path, processor)
            if result:
                all_results[file_path] = result
        
        # Summary comparison across files
        if len(all_results) > 1:
            print("\n" + "="*80)
            print("📊 CROSS-FILE COMPARISON")
            print("="*80)
            
            for file_path, results in all_results.items():
                print(f"\n📄 {Path(file_path).name}:")
                for strategy, data in results.items():
                    consistency = calculate_consistency(data['chunk_sizes'])
                    print(f"   {strategy:>12}: {data['chunk_count']:>3} chunks, "
                          f"avg {data['avg_size']:>6.1f} chars, {consistency}")
    
    # Test sample texts if no files provided or if requested
    if not args.files or args.strategy == 'all':
        print("\n📝 TESTING SAMPLE TEXTS")
        print("=" * 40)
        test_chunking_strategies()
    
    # Final recommendations
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS")
    print("="*80)
    print("✅ Recursive chunking: Best for structured documents (PDFs, technical docs)")
    print("✅ Sentence chunking: Good for narrative content (articles, stories)")
    print("✅ Adaptive chunking: Automatically chooses the best strategy per document")
    print("\n🔧 Configuration tips:")
    print("   - Increase chunk size for longer context")
    print("   - Increase overlap for better continuity")
    print("   - Use adaptive chunking for mixed document types")

if __name__ == "__main__":
    main()
