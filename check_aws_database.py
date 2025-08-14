#!/usr/bin/env python3
"""
Check the AWS PostgreSQL database for indexed documents
"""

import os
import psycopg2
import json
from psycopg2.extras import RealDictCursor

def check_database():
    """Check what documents are in the AWS PostgreSQL database"""
    
    # Database connection parameters
    db_params = {
        'host': os.environ.get('POSTGRES_HOST', 'localhost'),
        'port': 5432,
        'database': 'postgres',
        'user': 'postgres',
        'password': 'Vidyag09#'
    }
    
    try:
        print("🔌 Connecting to AWS PostgreSQL database...")
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Check if document_chunks table exists
        print("\n📋 Checking if document_chunks table exists...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'document_chunks'
            );
        """)
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print("❌ document_chunks table does not exist!")
            return
            
        print("✅ document_chunks table exists")
        
        # Get total count of documents
        print("\n📊 Checking total document chunks...")
        cursor.execute("SELECT COUNT(*) as total_chunks FROM document_chunks;")
        total_chunks = cursor.fetchone()['total_chunks']
        print(f"📈 Total chunks in database: {total_chunks}")
        
        if total_chunks == 0:
            print("❌ No document chunks found in the database!")
            return
        
        # Get unique documents
        print("\n📚 Checking unique documents...")
        cursor.execute("""
            SELECT s3_key, COUNT(*) as chunk_count, 
                   MIN(created_at) as first_indexed,
                   MAX(updated_at) as last_updated
            FROM document_chunks 
            GROUP BY s3_key 
            ORDER BY first_indexed DESC;
        """)
        documents = cursor.fetchall()
        
        print(f"📖 Found {len(documents)} unique documents:")
        for doc in documents:
            print(f"   📄 {doc['s3_key']} ({doc['chunk_count']} chunks)")
            print(f"       First indexed: {doc['first_indexed']}")
            print(f"       Last updated: {doc['last_updated']}")
            print()
        
        # Search for camp-related content
        print("\n🔍 Searching for camp-related content...")
        cursor.execute("""
            SELECT s3_key, chunk_index, 
                   LEFT(content, 200) as content_preview,
                   metadata
            FROM document_chunks 
            WHERE LOWER(content) LIKE '%camp%' 
               OR LOWER(content) LIKE '%director%'
               OR LOWER(s3_key) LIKE '%camp%'
               OR LOWER(s3_key) LIKE '%guide%'
            ORDER BY s3_key, chunk_index
            LIMIT 10;
        """)
        camp_chunks = cursor.fetchall()
        
        if camp_chunks:
            print(f"🏕️ Found {len(camp_chunks)} camp-related chunks:")
            for chunk in camp_chunks:
                print(f"\n📄 File: {chunk['s3_key']}")
                print(f"   Chunk {chunk['chunk_index']}: {chunk['content_preview']}...")
                if chunk['metadata']:
                    metadata = chunk['metadata'] if isinstance(chunk['metadata'], dict) else {}
                    print(f"   Metadata: {metadata}")
        else:
            print("❌ No camp-related content found!")
            
        # Test full-text search capability
        print("\n🔍 Testing full-text search for 'camp director'...")
        cursor.execute("""
            SELECT s3_key, chunk_index,
                   ts_rank(to_tsvector('english', content), plainto_tsquery('english', 'camp director')) as rank,
                   LEFT(content, 300) as content_preview
            FROM document_chunks 
            WHERE to_tsvector('english', content) @@ plainto_tsquery('english', 'camp director')
            ORDER BY rank DESC
            LIMIT 5;
        """)
        search_results = cursor.fetchall()
        
        if search_results:
            print(f"✅ Full-text search found {len(search_results)} results:")
            for result in search_results:
                print(f"\n📄 {result['s3_key']} (chunk {result['chunk_index']}) - Rank: {result['rank']:.4f}")
                print(f"   Content: {result['content_preview']}...")
        else:
            print("❌ Full-text search found no results for 'camp director'")
        
        cursor.close()
        conn.close()
        print("\n✅ Database check completed!")
        
    except psycopg2.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_database()
