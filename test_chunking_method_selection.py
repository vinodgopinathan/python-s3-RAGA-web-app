#!/usr/bin/env python3
"""
Test script to verify agentic chunking method selection
"""
import requests
import json
import time

# API configuration
API_BASE_URL = "http://localhost:5001"

def test_agentic_chunking():
    """Test that agentic chunking is properly applied when selected"""
    
    # Test data - using a complex document that should trigger agentic chunking
    test_text = """
    ARTIFICIAL INTELLIGENCE RESEARCH METHODOLOGY AND IMPLEMENTATION FRAMEWORK

    Abstract: This research paper presents a comprehensive methodology for implementing artificial intelligence systems in enterprise environments, analyzing the correlation between various machine learning algorithms and their statistical significance in real-world applications.

    1. Introduction
    The implementation of artificial intelligence (AI) frameworks requires a systematic approach to algorithm selection, data preprocessing, and model validation. According to recent studies (Smith et al., 2023), the effectiveness of AI systems depends significantly on the methodology employed during the development phase.

    2. Technical Architecture and Protocol Specifications
    2.1 Database Schema Design
    The proposed architecture utilizes a multi-tier database schema with the following specifications:
    - Primary tables for entity storage
    - Secondary indices for performance optimization  
    - Foreign key constraints for data integrity
    - JSON document storage for flexible metadata

    2.2 API Endpoint Configuration
    REST API endpoints must comply with the following requirements:
    - HTTP/HTTPS protocol support
    - JSON/XML data format compatibility
    - Authentication token validation
    - Rate limiting implementation

    3. Legal and Compliance Framework
    Whereas the implementation of AI systems must comply with relevant regulations, the following provisions shall be considered:

    3.1 Data Privacy Requirements
    All parties must ensure that data processing activities comply with applicable privacy laws. The organization shall implement appropriate technical and organizational measures to protect personal data pursuant to Article 32 of the GDPR.

    3.2 Liability and Indemnification
    Each party agrees to indemnify and hold harmless the other party from any claims arising from the use of the AI system, notwithstanding any limitations set forth herein.
    """
    
    # Test with different chunking methods
    methods_to_test = ['agentic', 'adaptive', 'semantic', 'recursive']
    
    for method in methods_to_test:
        print(f"\n🧪 Testing chunking method: {method}")
        
        # Prepare test request
        payload = {
            'directory_path': 'test_docs/',  # Use existing test directory
            'chunking_method': method,
            'chunk_size': 1000,
            'chunk_overlap': 200,
            'recursive': True
        }
        
        try:
            # Send request to process directory
            response = requests.post(
                f"{API_BASE_URL}/api/process-s3-directory",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Method {method}: Status = {result.get('status', 'unknown')}")
                print(f"   Job ID: {result.get('job_id', 'N/A')}")
                
                # Wait a moment for processing
                time.sleep(2)
                
                # Check job status to see actual chunking method used
                if 'job_id' in result:
                    status_response = requests.get(f"{API_BASE_URL}/api/job-status/{result['job_id']}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        print(f"   Job Status: {status_data.get('status', 'unknown')}")
                        print(f"   Chunking Method Used: {status_data.get('chunking_method', 'unknown')}")
                        
                        # Check documents processed
                        documents = status_data.get('result', {}).get('documents', [])
                        if documents:
                            print(f"   Documents processed: {len(documents)}")
                            for doc in documents[:2]:  # Show first 2 documents
                                print(f"     - {doc.get('file_path', 'unknown')}: {doc.get('chunks_created', 0)} chunks")
                    else:
                        print(f"   ❌ Failed to get job status: {status_response.status_code}")
            else:
                print(f"❌ Method {method}: Failed with status {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Method {method}: Exception occurred: {e}")
        
        print("-" * 50)

def test_chunking_methods_endpoint():
    """Test the chunking methods endpoint"""
    print("\n🔍 Testing chunking methods endpoint...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/chunking-methods")
        if response.status_code == 200:
            data = response.json()
            methods = data.get('methods', [])
            print(f"✅ Available chunking methods: {len(methods)}")
            for method in methods:
                print(f"   - {method.get('name', 'unknown')}: {method.get('description', 'No description')}")
        else:
            print(f"❌ Failed to get chunking methods: {response.status_code}")
    except Exception as e:
        print(f"❌ Exception getting chunking methods: {e}")

if __name__ == "__main__":
    print("🚀 Testing Chunking Method Selection")
    print("=" * 60)
    
    # First test available methods
    test_chunking_methods_endpoint()
    
    # Then test actual chunking with different methods
    test_agentic_chunking()
    
    print("\n✨ Test completed!")
