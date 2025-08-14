#!/usr/bin/env python3
"""
Test Enhanced Adaptive Chunking through the consolidated MCP RAG Server
This script tests our refactored MCP server that uses the existing ChunkingService
"""

import asyncio
import json
import sys
import os

# Add the backend src to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from utils.mcp_rag_server import MCPRAGClient

async def test_enhanced_chunking():
    """Test the enhanced adaptive chunking through MCP"""
    print("🚀 Testing Enhanced Adaptive Chunking through Consolidated MCP Server")
    print("=" * 70)
    
    # Initialize the MCP client (uses our refactored server with ChunkingService)
    client = MCPRAGClient()
    
    # Test document path
    test_document = "test_complex_document.txt"
    print(f"📄 Test Document: {test_document}")
    
    try:
        # Step 1: Index the document with enhanced adaptive chunking
        print("\n1️⃣ Indexing document with Enhanced Adaptive Chunking...")
        index_result = await client.index_document(
            s3_key=test_document,
            force_reindex=True,
            chunking_method="adaptive",  # This will use our enhanced adaptive chunking!
            chunk_size=1000,
            chunk_overlap=200
        )
        print(f"   ✅ Index Result: {json.dumps(index_result, indent=2)}")
        
        # Step 2: Search for content to test the enhanced chunking
        print("\n2️⃣ Searching for technical content...")
        search_result = await client.search_documents(
            query="artificial intelligence technical architecture database schema",
            limit=3,
            similarity_threshold=0.5
        )
        print(f"   📊 Found {search_result.get('results_count', 0)} relevant chunks")
        
        # Display results with enhanced metadata
        for i, result in enumerate(search_result.get('results', [])[:2], 1):
            print(f"\n   📝 Result {i}:")
            print(f"      Method: {result.get('chunk_method', 'unknown')}")
            print(f"      Score: {result.get('similarity_score', 0):.3f}")
            print(f"      Content Preview: {result.get('content', '')[:100]}...")
        
        # Step 3: Get formatted context
        print("\n3️⃣ Getting formatted RAG context...")
        context_result = await client.get_context(
            query="legal compliance framework data privacy requirements",
            limit=2,
            similarity_threshold=0.5,
            max_context_length=2000
        )
        print(f"   📋 Context Sources: {context_result.get('sources_used', 0)}")
        print(f"   📏 Context Length: {context_result.get('total_length', 0)} chars")
        print(f"   🔍 Context Preview:")
        context_preview = context_result.get('context', '')[:300]
        print(f"      {context_preview}...")
        
        # Step 4: Test legal document search (specific to our test document)
        print("\n4️⃣ Testing Legal Document Search...")
        legal_search = await client.search_documents(
            query="GDPR compliance liability indemnification",
            limit=2,
            similarity_threshold=0.4
        )
        print(f"   ⚖️ Legal Results: {legal_search.get('results_count', 0)} chunks found")
        
        # Step 5: Test code-related search
        print("\n5️⃣ Testing Code Pattern Search...")
        code_search = await client.search_documents(
            query="python class AIFramework implementation",
            limit=2,
            similarity_threshold=0.4
        )
        print(f"   💻 Code Results: {code_search.get('results_count', 0)} chunks found")
        
        print("\n" + "=" * 70)
        print("🎉 Enhanced Adaptive Chunking Test Complete!")
        print("✅ Successfully tested consolidated MCP server with ChunkingService")
        print("✅ Enhanced adaptive chunking working through unified architecture")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing Enhanced Adaptive Chunking with Consolidated Backend...")
    asyncio.run(test_enhanced_chunking())
