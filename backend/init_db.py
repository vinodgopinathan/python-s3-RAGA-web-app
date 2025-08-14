#!/usr/bin/env python3
"""
Database initialization script for RAG vector database
This script sets up the PostgreSQL database with pgvector extension
and creates the necessary tables for document storage.
"""

import os
import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'localhost'),
        port=os.environ.get('POSTGRES_PORT', '5432'),
        user=os.environ.get('POSTGRES_USER', 'postgres'),
        password=os.environ.get('POSTGRES_PASSWORD'),
        database=os.environ.get('POSTGRES_DB', 'postgres')
    )

def init_database():
    """Initialize the database with required extensions and tables"""
    try:
        logger.info("Connecting to PostgreSQL database...")
        conn = get_db_connection()
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Enable pgvector extension
        logger.info("Enabling pgvector extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        logger.info("pgvector extension enabled successfully")
        
        # Check if we can query vector functions
        cursor.execute("SELECT 1;")
        logger.info("Database connection verified")
        
        # Get embedding dimension (384 for all-MiniLM-L6-v2)
        embedding_dimension = 384
        
        # Create document_chunks table if it doesn't exist
        logger.info("Creating document_chunks table...")
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            s3_key TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            metadata JSONB DEFAULT '{{}}'::jsonb,
            embedding vector({embedding_dimension}),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_table_query)
        logger.info("document_chunks table created successfully")
        
        # Create indexes for better performance
        logger.info("Creating indexes...")
        
        # Index on s3_key for document lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_s3_key 
            ON document_chunks(s3_key);
        """)
        
        # Vector similarity index (ivfflat)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding 
            ON document_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
        # Index on created_at for temporal queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_chunks_created_at 
            ON document_chunks(created_at);
        """)
        
        logger.info("Indexes created successfully")
        
        # Create a function to update the updated_at timestamp
        logger.info("Creating update timestamp function...")
        cursor.execute("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = CURRENT_TIMESTAMP;
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        # Create trigger for automatic timestamp updates
        cursor.execute("""
            DROP TRIGGER IF EXISTS update_document_chunks_updated_at ON document_chunks;
            CREATE TRIGGER update_document_chunks_updated_at
                BEFORE UPDATE ON document_chunks
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        """)
        
        logger.info("Timestamp function and trigger created successfully")
        
        # Verify table exists and show structure
        cursor.execute("""
            SELECT column_name, data_type, character_maximum_length, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'document_chunks'
            ORDER BY ordinal_position;
        """)
        
        columns = cursor.fetchall()
        logger.info("Table structure:")
        for col in columns:
            logger.info(f"  {col[0]} ({col[1]}) - Nullable: {col[3]}")
        
        # Show current row count
        cursor.execute("SELECT COUNT(*) FROM document_chunks;")
        row_count = cursor.fetchone()[0]
        logger.info(f"Current document chunks count: {row_count}")
        
        logger.info("Database initialization completed successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def test_database():
    """Test database connection and basic operations"""
    try:
        logger.info("Testing database connection...")
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        logger.info(f"PostgreSQL version: {version}")
        
        # Test pgvector extension
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cursor.fetchone()
        if vector_ext:
            logger.info("pgvector extension is installed")
        else:
            logger.warning("pgvector extension is not installed")
        
        # Test table access
        cursor.execute("SELECT COUNT(*) FROM document_chunks;")
        count = cursor.fetchone()[0]
        logger.info(f"Document chunks table accessible with {count} rows")
        
        logger.info("Database test completed successfully!")
        
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_database()
    else:
        init_database()

if __name__ == "__main__":
    main()
