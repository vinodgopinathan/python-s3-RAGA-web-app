#!/usr/bin/env python3
"""
Test script to verify that adaptive chunking now stores the actual method used
"""

import psycopg2
import os
import json
from typing import Dict, Any

def get_db_connection():
    """Get database connection using environment variables"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        database=os.getenv('DB_NAME', 'rag_db'),
        user=os.getenv('DB_USER', 'rag_user'),
        password=os.getenv('DB_PASSWORD', 'rag_password')
    )

def test_chunking_methods():
    """Test and display chunking methods from recent documents"""
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        print("=== Testing Chunking Methods ===\n")
        
        # Get recent documents with their chunking information
        cursor.execute("""
            SELECT 
                id,
                s3_key,
                chunking_method,
                metadata
            FROM documents 
            WHERE status = 'processed' 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        
        documents = cursor.fetchall()
        
        if not documents:
            print("No processed documents found in database.")
            return
        
        print(f"Found {len(documents)} recent processed documents:\n")
        
        for doc_id, s3_key, chunking_method, metadata in documents:
            print(f"Document ID: {doc_id}")
            print(f"S3 Key: {s3_key}")
            print(f"Chunking Method: {chunking_method}")
            
            # Parse metadata if it's a JSON string
            try:
                if isinstance(metadata, str):
                    metadata_obj = json.loads(metadata)
                else:
                    metadata_obj = metadata or {}
                
                # Display relevant metadata fields
                if 'adaptive_method' in metadata_obj:
                    print(f"  Adaptive Method: {metadata_obj['adaptive_method']}")
                if 'original_request' in metadata_obj:
                    print(f"  Original Request: {metadata_obj['original_request']}")
                if 'adaptive_reasoning' in metadata_obj:
                    print(f"  Adaptive Reasoning: {metadata_obj['adaptive_reasoning']}")
                    
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  Metadata parsing error: {e}")
            
            print("-" * 50)
        
        # Check chunks for more detailed method information
        print("\n=== Checking Chunks for Method Details ===\n")
        
        cursor.execute("""
            SELECT 
                c.document_id,
                d.s3_key,
                c.metadata
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.status = 'processed'
            AND c.metadata IS NOT NULL
            ORDER BY c.document_id DESC, c.chunk_index ASC
            LIMIT 5
        """)
        
        chunks = cursor.fetchall()
        
        for doc_id, s3_key, chunk_metadata in chunks:
            print(f"Document: {s3_key} (ID: {doc_id})")
            
            try:
                if isinstance(chunk_metadata, str):
                    metadata_obj = json.loads(chunk_metadata)
                else:
                    metadata_obj = chunk_metadata or {}
                
                chunking_method = metadata_obj.get('chunking_method', 'unknown')
                adaptive_method = metadata_obj.get('adaptive_method', 'N/A')
                original_request = metadata_obj.get('original_request', 'N/A')
                
                print(f"  Chunking Method: {chunking_method}")
                if adaptive_method != 'N/A':
                    print(f"  Adaptive Method: {adaptive_method}")
                if original_request != 'N/A':
                    print(f"  Original Request: {original_request}")
                    
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  Chunk metadata parsing error: {e}")
            
            print("-" * 30)
            break  # Just show one chunk per document
        
        # Summary statistics
        print("\n=== Chunking Method Summary ===\n")
        
        cursor.execute("""
            SELECT 
                chunking_method,
                COUNT(*) as count
            FROM documents 
            WHERE status = 'processed' 
            GROUP BY chunking_method
            ORDER BY count DESC
        """)
        
        method_counts = cursor.fetchall()
        
        for method, count in method_counts:
            print(f"{method}: {count} documents")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error testing chunking methods: {e}")

if __name__ == "__main__":
    test_chunking_methods()
