#!/usr/bin/env python3
"""
Test script to debug chunking method selection
"""
import os
import sys
sys.path.append('./backend/src')

from utils.document_processor import DocumentProcessor

# Set up environment for testing
os.environ['CHUNKING_METHOD'] = 'adaptive'
os.environ['LLM_PROVIDER'] = 'gemini' 
os.environ['GEMINI_API_KEY'] = 'AIzaSyApqc6-5rx8bNEsBQHL6fnhpLKP6RgfgpU'
os.environ['CHUNK_SIZE'] = '1000'
os.environ['CHUNK_OVERLAP'] = '200'

# Read test document
with open('./test_docs/test_complex_document.txt', 'r') as f:
    test_text = f.read()

# Initialize document processor
processor = DocumentProcessor()

# Analyze text characteristics
text_stats = processor._analyze_text_characteristics(test_text)
print("Text Analysis:")
for key, value in text_stats.items():
    print(f"  {key}: {value}")

# Get method selection
selected_method = processor._select_optimal_chunking_method(text_stats, test_text)
print(f"\nSelected Method: {selected_method}")

# Test chunking with adaptive method
chunks = processor.chunk_text(test_text)
print(f"\nNumber of chunks: {len(chunks)}")
if chunks:
    print(f"Actual chunking method used: {chunks[0]['metadata'].get('chunking_method', 'unknown')}")
    print(f"First chunk (first 200 chars): {chunks[0]['content'][:200]}...")
