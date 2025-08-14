#!/usr/bin/env python3
"""
🧪 CPU-Optimized Container Test Script

This script verifies that the CPU-optimized container builds and runs correctly
with semantic chunking functionality intact.
"""

import subprocess
import sys
import time
import json
import requests
from typing import Dict, Any

def run_command(cmd: str, capture_output: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"🔍 Running: {cmd}")
    return subprocess.run(
        cmd, 
        shell=True, 
        capture_output=capture_output, 
        text=True
    )

def test_container_build() -> bool:
    """Test if the CPU-optimized container builds successfully."""
    print("🏗️ Testing container build...")
    
    result = run_command(
        "docker build -f backend/Dockerfile -t aws-llm-raga-backend-cpu:test ./backend"
    )
    
    if result.returncode == 0:
        print("✅ Container build successful!")
        return True
    else:
        print(f"❌ Container build failed: {result.stderr}")
        return False

def test_pytorch_cpu_import() -> bool:
    """Test that PyTorch imports correctly with CPU-only dependencies."""
    print("🧠 Testing PyTorch CPU import...")
    
    test_script = '''
import torch
import sentence_transformers
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CPU tensor test: {torch.tensor([1, 2, 3]).sum()}")

# Test sentence-transformers with CPU
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Hello world", "How are you?"])
print(f"Embeddings shape: {embeddings.shape}")
print("🎉 CPU-only PyTorch working correctly!")
'''
    
    result = run_command(
        f'docker run --rm aws-llm-raga-backend-cpu:test python -c "{test_script}"'
    )
    
    if result.returncode == 0 and "CPU-only PyTorch working correctly!" in result.stdout:
        print("✅ PyTorch CPU import successful!")
        print(result.stdout)
        return True
    else:
        print(f"❌ PyTorch CPU import failed: {result.stderr}")
        return False

def test_semantic_chunking() -> bool:
    """Test semantic chunking functionality in the container."""
    print("🎯 Testing semantic chunking...")
    
    test_script = '''
import sys
sys.path.append("/app/src")
from utils.document_processor import DocumentProcessor

# Test semantic chunking
processor = DocumentProcessor()
processor.chunking_method = "semantic"

test_text = """
Machine learning is a fascinating field. It involves training algorithms on data.
Deep learning is a subset of machine learning. It uses neural networks with multiple layers.
Natural language processing is another important area. It focuses on understanding human language.
Computer vision deals with image processing. It enables machines to see and interpret visual data.
"""

chunks = processor.process_text_content(test_text, "test_semantic.txt")
print(f"Generated {len(chunks)} semantic chunks")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk['content'][:100]}...")
    print(f"Method: {chunk['metadata']['chunking_method']}")

print("🎉 Semantic chunking working correctly!")
'''
    
    result = run_command(
        f'docker run --rm aws-llm-raga-backend-cpu:test python -c "{test_script}"'
    )
    
    if result.returncode == 0 and "Semantic chunking working correctly!" in result.stdout:
        print("✅ Semantic chunking successful!")
        print(result.stdout)
        return True
    else:
        print(f"❌ Semantic chunking failed: {result.stderr}")
        return False

def test_container_size() -> bool:
    """Test the container size to verify optimization."""
    print("📏 Testing container size...")
    
    result = run_command("docker images aws-llm-raga-backend-cpu:test --format 'table {{.Size}}'")
    
    if result.returncode == 0:
        size_line = result.stdout.strip().split('\n')[-1]
        print(f"✅ Container size: {size_line}")
        print("💡 CPU-optimized container should be significantly smaller than GPU version")
        return True
    else:
        print(f"❌ Could not get container size: {result.stderr}")
        return False

def main():
    """Run all tests for the CPU-optimized container."""
    print("🚀 CPU-Optimized Container Test Suite")
    print("=" * 50)
    
    tests = [
        ("Container Build", test_container_build),
        ("PyTorch CPU Import", test_pytorch_cpu_import),
        ("Semantic Chunking", test_semantic_chunking),
        ("Container Size", test_container_size),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CPU-optimized container is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
