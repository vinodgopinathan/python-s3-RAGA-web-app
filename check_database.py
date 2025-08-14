#!/usr/bin/env python3

import os
import psycopg2
from psycopg2.extras import RealDictCursor

def check_camper_guidebook_chunks():
    """Check if Camper Guidebook chunks exist in the database and search for Kellie Cook"""
    
    # Database connection parameters
    db_params = {
        'host': os.environ.get('DATABASE_HOST', 'localhost'),
        'port': os.environ.get('DATABASE_PORT', 5432),
        'database': os.environ.get('DATABASE_NAME', 'aws_llm_db'),
        'user': os.environ.get('DATABASE_USER', 'aws_llm_user'),
        'password': os.environ.get('DATABASE_PASSWORD', 'aws_llm_password')
    }
    
    try:
        # Connect to database
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        print("=== CHECKING CAMPER GUIDEBOOK CHUNKS ===")
        
        # Check if any chunks exist for Camper Guidebook
        cursor.execute("""
            SELECT COUNT(*) as chunk_count 
            FROM document_chunks 
            WHERE s3_key ILIKE '%camper%guidebook%'
        """)
        
        result = cursor.fetchone()
        print(f"Total chunks for Camper Guidebook: {result['chunk_count']}")
        
        if result['chunk_count'] > 0:
            # Get some sample chunks
            cursor.execute("""
                SELECT id, s3_key, chunk_index, 
                       LEFT(content, 100) as content_preview,
                       LENGTH(content) as content_length
                FROM document_chunks 
                WHERE s3_key ILIKE '%camper%guidebook%'
                ORDER BY chunk_index
                LIMIT 10
            """)
            
            chunks = cursor.fetchall()
            print(f"\nSample chunks:")
            for chunk in chunks:
                print(f"  ID: {chunk['id']}, Index: {chunk['chunk_index']}, Length: {chunk['content_length']}")
                print(f"  Preview: {chunk['content_preview']}...")
                print()
            
            # Search for "Kellie Cook" in all chunks
            print("=== SEARCHING FOR 'KELLIE COOK' ===")
            cursor.execute("""
                SELECT id, s3_key, chunk_index, content,
                       LENGTH(content) as content_length
                FROM document_chunks 
                WHERE s3_key ILIKE '%camper%guidebook%'
                AND (content ILIKE '%kellie%cook%' OR content ILIKE '%cook%kellie%')
            """)
            
            kellie_chunks = cursor.fetchall()
            print(f"Chunks containing 'Kellie Cook': {len(kellie_chunks)}")
            
            if kellie_chunks:
                for chunk in kellie_chunks:
                    print(f"\nFOUND in chunk {chunk['chunk_index']}:")
                    print(f"Content: {chunk['content']}")
                    print("-" * 80)
            
            # Search for just "Kellie" 
            cursor.execute("""
                SELECT id, s3_key, chunk_index, content,
                       LENGTH(content) as content_length
                FROM document_chunks 
                WHERE s3_key ILIKE '%camper%guidebook%'
                AND content ILIKE '%kellie%'
            """)
            
            kellie_only_chunks = cursor.fetchall()
            print(f"\nChunks containing just 'Kellie': {len(kellie_only_chunks)}")
            
            if kellie_only_chunks:
                for chunk in kellie_only_chunks:
                    print(f"\nFOUND 'Kellie' in chunk {chunk['chunk_index']}:")
                    # Find the position of "kellie" and show context
                    content = chunk['content']
                    kellie_pos = content.lower().find('kellie')
                    if kellie_pos >= 0:
                        start = max(0, kellie_pos - 50)
                        end = min(len(content), kellie_pos + 100)
                        context = content[start:end]
                        print(f"Context: ...{context}...")
                    print("-" * 80)
            
            # Search for just "Cook"
            cursor.execute("""
                SELECT id, s3_key, chunk_index, content,
                       LENGTH(content) as content_length
                FROM document_chunks 
                WHERE s3_key ILIKE '%camper%guidebook%'
                AND content ILIKE '%cook%'
            """)
            
            cook_chunks = cursor.fetchall()
            print(f"\nChunks containing 'Cook': {len(cook_chunks)}")
            
            if cook_chunks:
                for chunk in cook_chunks:
                    print(f"\nFOUND 'Cook' in chunk {chunk['chunk_index']}:")
                    # Find the position of "cook" and show context
                    content = chunk['content']
                    cook_pos = content.lower().find('cook')
                    if cook_pos >= 0:
                        start = max(0, cook_pos - 50)
                        end = min(len(content), cook_pos + 100)
                        context = content[start:end]
                        print(f"Context: ...{context}...")
                    print("-" * 80)
        else:
            print("No chunks found for Camper Guidebook - it might not be indexed yet.")
            
            # Check what documents ARE indexed
            cursor.execute("""
                SELECT s3_key, COUNT(*) as chunk_count
                FROM document_chunks 
                GROUP BY s3_key
                ORDER BY chunk_count DESC
                LIMIT 10
            """)
            
            docs = cursor.fetchall()
            print(f"\nTop indexed documents:")
            for doc in docs:
                print(f"  {doc['s3_key']}: {doc['chunk_count']} chunks")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Database error: {e}")

if __name__ == "__main__":
    check_camper_guidebook_chunks()
