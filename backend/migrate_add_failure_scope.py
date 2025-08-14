#!/usr/bin/env python3
"""
Migration script to add failure_scope column to processing_errors table
and populate it based on existing data.
"""

import os
import sys
import psycopg2
import psycopg2.extras
import logging

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get database connection using environment variables"""
    try:
        return psycopg2.connect(
            host=os.environ.get('DB_HOST') or os.environ.get('POSTGRES_HOST', 'localhost'),
            port=int(os.environ.get('DB_PORT') or os.environ.get('POSTGRES_PORT', '5432')),
            database=os.environ.get('DB_NAME') or os.environ.get('POSTGRES_DB', 'rag_system'),
            user=os.environ.get('DB_USER') or os.environ.get('POSTGRES_USER', 'postgres'),
            password=os.environ.get('DB_PASSWORD') or os.environ.get('POSTGRES_PASSWORD', '')
        )
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

def migrate_add_failure_scope():
    """Add failure_scope column and populate it"""
    
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # Check if column already exists
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'processing_errors' 
                AND column_name = 'failure_scope'
            """)
            
            column_exists = cur.fetchone() is not None
            
            if not column_exists:
                logger.info("Adding failure_scope column to processing_errors table...")
                
                # Add the column with default value
                cur.execute("""
                    ALTER TABLE processing_errors 
                    ADD COLUMN failure_scope TEXT NOT NULL DEFAULT 'document' 
                    CHECK (failure_scope IN ('document', 'chunk'))
                """)
                
                logger.info("Column added successfully")
            else:
                logger.info("failure_scope column already exists")
            
            # Update existing records based on error_details content
            logger.info("Updating existing records to set appropriate failure_scope...")
            
            # Set chunk scope for records that have chunk-specific error details
            cur.execute("""
                UPDATE processing_errors 
                SET failure_scope = 'chunk'
                WHERE error_details ? 'chunk_error_type' 
                   OR error_details ? 'failed_chunk_count'
                   OR error_details ? 'failed_chunk_indices'
            """)
            
            chunk_updates = cur.rowcount
            logger.info(f"Updated {chunk_updates} records to 'chunk' scope")
            
            # Verify document scope records (should already be 'document' due to default)
            cur.execute("""
                SELECT COUNT(*) 
                FROM processing_errors 
                WHERE failure_scope = 'document'
            """)
            
            document_count = cur.fetchone()[0]
            logger.info(f"Found {document_count} records with 'document' scope")
            
            # Create index for the new column
            logger.info("Creating indexes for failure_scope...")
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_processing_errors_failure_scope 
                ON processing_errors(failure_scope)
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_processing_errors_scope_type 
                ON processing_errors(failure_scope, error_type)
            """)
            
            logger.info("Indexes created successfully")
            
            conn.commit()
            
            # Print summary
            cur.execute("""
                SELECT 
                    failure_scope,
                    COUNT(*) as count,
                    COUNT(DISTINCT error_type) as distinct_error_types,
                    MIN(created_at) as earliest_error,
                    MAX(created_at) as latest_error
                FROM processing_errors 
                GROUP BY failure_scope
                ORDER BY failure_scope
            """)
            
            logger.info("Migration summary:")
            for row in cur.fetchall():
                scope, count, types, earliest, latest = row
                logger.info(f"  {scope}: {count} errors, {types} error types, {earliest} to {latest}")
            
            logger.info("Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_add_failure_scope()
