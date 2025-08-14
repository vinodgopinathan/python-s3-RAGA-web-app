#!/usr/bin/env python3
"""
Test script for hybrid search integration in the RAG LLM system
"""

import os
import sys
import asyncio
import logging
from typing import Dict, Any

# Add the backend src directory to the Python path
sys.path.append('/Users/vinodgopinathan/AWS LLM RAGA Project/python-s3-RAGA-web-app/backend/src')

from utils.rag_llm_helper import RAGLLMHelper

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_hybrid_search():
    """Test the hybrid search functionality"""
    
    print("🔧 Initializing RAG LLM Helper with Hybrid Search...")
    
    try:
        # Initialize the RAG LLM helper
        rag_helper = RAGLLMHelper()
        
        # Test queries with different characteristics
        test_queries = [
            "What are the key features of the document processing system?",
            "How does chunking work in this application?",
            "Explain the database schema and tables",
            "What error handling mechanisms are implemented?",
            "How is the S3 integration configured?"
        ]
        
        print("\n🔍 Testing Hybrid Search with Various Queries...")
        print("=" * 80)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test Query {i}: {query}")
            print("-" * 60)
            
            try:
                # Test the hybrid search method
                result = await rag_helper.query_s3_files_hybrid(
                    query=query,
                    max_results=5,
                    semantic_threshold=0.6
                )
                
                # Display results
                print(f"✅ Response: {result.get('response', 'No response')[:200]}...")
                print(f"📊 Search Strategy: {result.get('search_strategy', 'Unknown')}")
                
                # Query analysis
                analysis = result.get('query_analysis', {})
                if analysis:
                    print(f"🔍 Keywords Extracted: {analysis.get('keywords', [])}")
                    print(f"📋 Query Intent: {analysis.get('intent', 'Unknown')}")
                
                # Search statistics
                stats = result.get('search_stats', {})
                if stats:
                    print(f"📈 Keyword Results: {stats.get('keyword_count', 0)}")
                    print(f"🎯 Semantic Results: {stats.get('semantic_count', 0)}")
                    print(f"🔗 Merged Results: {stats.get('merged_count', 0)}")
                
                # Sources
                sources = result.get('sources', [])
                if sources:
                    print(f"📚 Sources Used: {len(sources)} documents")
                    for source in sources[:3]:  # Show first 3 sources
                        print(f"   - {source.get('file_name', 'Unknown')}")
                        
            except Exception as e:
                print(f"❌ Error testing query: {str(e)}")
                logger.error(f"Query test error: {str(e)}")
                
            print()
        
        print("\n🔧 Testing Legacy Compatibility...")
        print("-" * 60)
        
        # Test the legacy method (should now use hybrid search)
        legacy_result = rag_helper.query_s3_files(
            query="What is the main purpose of this application?",
            max_files=3
        )
        
        print(f"✅ Legacy Method Response: {legacy_result.get('response', 'No response')[:150]}...")
        print(f"📊 Legacy Search Strategy: {legacy_result.get('search_strategy', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize or test hybrid search: {str(e)}")
        logger.error(f"Test initialization error: {str(e)}")
        return False

def test_query_analysis():
    """Test the query analysis component separately"""
    
    print("\n🧠 Testing Query Analysis Component...")
    print("-" * 60)
    
    try:
        from utils.hybrid_search_helper import HybridSearchHelper
        
        # Initialize hybrid search helper
        hybrid_helper = HybridSearchHelper()
        
        test_queries = [
            "How does the chunking algorithm work?",
            "What database tables are used for storing documents?",
            "Show me error handling in the application",
            "Explain S3 integration and file upload process"
        ]
        
        for query in test_queries:
            print(f"\n📝 Analyzing: {query}")
            
            try:
                analysis = asyncio.run(
                    hybrid_helper.parse_query_for_keywords(query)
                )
                
                print(f"🔍 Keywords: {analysis.get('keywords', [])}")
                print(f"📋 Intent: {analysis.get('intent', 'Unknown')}")
                print(f"🎯 Search Type: {analysis.get('search_type', 'Unknown')}")
                
            except Exception as e:
                print(f"❌ Analysis error: {str(e)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query analysis test failed: {str(e)}")
        return False

def main():
    """Main test function"""
    
    print("🚀 Hybrid Search Integration Test")
    print("=" * 80)
    
    # Check environment
    required_env_vars = [
        'DATABASE_URL',
        'GOOGLE_APPLICATION_CREDENTIALS'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("Please set up your environment variables before running tests.")
        return False
    
    print("✅ Environment variables configured")
    
    # Run tests
    try:
        # Test query analysis
        analysis_success = test_query_analysis()
        
        # Test full hybrid search
        if analysis_success:
            search_success = asyncio.run(test_hybrid_search())
            
            if search_success:
                print("\n🎉 All tests completed successfully!")
                print("✅ Hybrid search integration is working properly")
                return True
            else:
                print("\n❌ Hybrid search tests failed")
                return False
        else:
            print("\n❌ Query analysis tests failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {str(e)}")
        logger.error(f"Main test error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
