#!/usr/bin/env python3
"""
Database Migration Script for Error Tracking
Adds the processing_errors table and related views to existing database
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    """Get database connection using environment variables"""
    try:
        # Try the standard DB_* variables first
        db_config = {
            'host': os.environ.get('DB_HOST') or os.environ.get('POSTGRES_HOST'),
            'port': int(os.environ.get('DB_PORT', os.environ.get('POSTGRES_PORT', 5432))),
            'database': os.environ.get('DB_NAME') or os.environ.get('POSTGRES_DB'),
            'user': os.environ.get('DB_USER') or os.environ.get('POSTGRES_USER'),
            'password': os.environ.get('DB_PASSWORD') or os.environ.get('POSTGRES_PASSWORD')
        }
        
        # Check if all required fields are present
        missing_fields = [k for k, v in db_config.items() if v is None]
        if missing_fields:
            raise ValueError(f"Missing database configuration: {missing_fields}")
        
        print(f"Connecting to database: {db_config['host']}:{db_config['port']}/{db_config['database']}")
        conn = psycopg2.connect(**db_config)
        return conn
        
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        print("\nPlease ensure the following environment variables are set:")
        print("- DB_HOST (or POSTGRES_HOST)")
        print("- DB_PORT (or POSTGRES_PORT)")
        print("- DB_NAME (or POSTGRES_DB)")
        print("- DB_USER (or POSTGRES_USER)")
        print("- DB_PASSWORD (or POSTGRES_PASSWORD)")
        sys.exit(1)

def apply_migration():
    """Apply the error tracking migration"""
    
    migration_sql = """
    -- Processing errors table to track all errors during document processing
    CREATE TABLE IF NOT EXISTS processing_errors (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
        file_name TEXT NOT NULL,
        file_path TEXT,
        s3_key TEXT,
        error_type TEXT NOT NULL CHECK (error_type IN ('file_access', 'password_protected', 'format_unsupported', 'parsing_error', 'chunking_error', 'embedding_error', 'database_error', 'network_error', 'validation_error', 'other')),
        error_message TEXT NOT NULL,
        error_details JSONB DEFAULT '{}', -- Store stack traces, additional context
        processing_stage TEXT NOT NULL CHECK (processing_stage IN ('file_download', 'file_extraction', 'text_chunking', 'embedding_generation', 'database_storage', 'validation')),
        retry_count INTEGER DEFAULT 0,
        is_recoverable BOOLEAN DEFAULT TRUE,
        resolved BOOLEAN DEFAULT FALSE,
        resolution_notes TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        resolved_at TIMESTAMP WITH TIME ZONE
    );

    -- Processing errors indexes
    CREATE INDEX IF NOT EXISTS idx_processing_errors_document_id ON processing_errors(document_id);
    CREATE INDEX IF NOT EXISTS idx_processing_errors_error_type ON processing_errors(error_type);
    CREATE INDEX IF NOT EXISTS idx_processing_errors_processing_stage ON processing_errors(processing_stage);
    CREATE INDEX IF NOT EXISTS idx_processing_errors_resolved ON processing_errors(resolved);
    CREATE INDEX IF NOT EXISTS idx_processing_errors_created_at ON processing_errors(created_at);
    CREATE INDEX IF NOT EXISTS idx_processing_errors_file_name ON processing_errors(file_name);

    -- Add trigger for timestamp updates
    DROP TRIGGER IF EXISTS update_processing_errors_updated_at ON processing_errors;
    CREATE TRIGGER update_processing_errors_updated_at 
        BEFORE UPDATE ON processing_errors 
        FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

    -- Error analytics view
    CREATE OR REPLACE VIEW error_analytics AS
    SELECT 
        error_type,
        processing_stage,
        COUNT(*) as error_count,
        COUNT(CASE WHEN resolved = true THEN 1 END) as resolved_count,
        COUNT(CASE WHEN resolved = false THEN 1 END) as unresolved_count,
        COUNT(CASE WHEN is_recoverable = true THEN 1 END) as recoverable_count,
        AVG(retry_count) as avg_retries,
        MIN(created_at) as first_occurrence,
        MAX(created_at) as last_occurrence
    FROM processing_errors 
    GROUP BY error_type, processing_stage
    ORDER BY error_count DESC;

    -- Documents with errors view
    CREATE OR REPLACE VIEW documents_with_errors AS
    SELECT 
        d.id,
        d.file_name,
        d.file_path,
        d.processing_status,
        d.error_message as document_error,
        COUNT(pe.id) as error_count,
        STRING_AGG(DISTINCT pe.error_type, ', ') as error_types,
        STRING_AGG(DISTINCT pe.processing_stage, ', ') as failed_stages,
        MAX(pe.created_at) as last_error_at
    FROM documents d
    LEFT JOIN processing_errors pe ON d.id = pe.document_id
    WHERE d.processing_status = 'failed' OR pe.id IS NOT NULL
    GROUP BY d.id, d.file_name, d.file_path, d.processing_status, d.error_message
    ORDER BY error_count DESC, last_error_at DESC;
    """
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            print("Applying error tracking migration...")
            cursor.execute(migration_sql)
            conn.commit()
            print("✅ Migration applied successfully!")
            
            # Test the new table
            cursor.execute("SELECT COUNT(*) FROM processing_errors;")
            count = cursor.fetchone()[0]
            print(f"✅ processing_errors table ready (current count: {count})")
            
            # Test the views
            cursor.execute("SELECT COUNT(*) FROM error_analytics;")
            analytics_count = cursor.fetchone()[0]
            print(f"✅ error_analytics view ready (current error types: {analytics_count})")
            
            cursor.execute("SELECT COUNT(*) FROM documents_with_errors;")
            doc_errors_count = cursor.fetchone()[0]
            print(f"✅ documents_with_errors view ready (documents with errors: {doc_errors_count})")
            
    except Exception as e:
        conn.rollback()
        print(f"❌ Migration failed: {e}")
        sys.exit(1)
    finally:
        conn.close()

def check_existing_tables():
    """Check which tables already exist"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Check for existing tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name;
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            print("Existing tables:")
            for table in tables:
                print(f"  - {table}")
            
            # Check if processing_errors already exists
            if 'processing_errors' in tables:
                cursor.execute("SELECT COUNT(*) FROM processing_errors;")
                count = cursor.fetchone()[0]
                print(f"\n⚠️  processing_errors table already exists with {count} records")
                
                response = input("Do you want to continue anyway? (y/N): ")
                if response.lower() != 'y':
                    print("Migration cancelled.")
                    sys.exit(0)
            else:
                print(f"\n✅ processing_errors table does not exist yet - safe to proceed")
                
    except Exception as e:
        print(f"Error checking existing tables: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    print("🚀 Database Migration: Error Tracking System")
    print("=" * 50)
    
    # Check current state
    check_existing_tables()
    
    print("\nApplying migration...")
    apply_migration()
    
    print("\n🎉 Migration completed successfully!")
    print("\nNew error tracking features available:")
    print("  - API endpoint: /api/errors/summary")
    print("  - API endpoint: /api/errors/password-protected")
    print("  - API endpoint: /api/errors/recoverable")
    print("  - API endpoint: /api/errors/document/<id>")
    print("  - API endpoint: /api/errors/<id>/resolve")
