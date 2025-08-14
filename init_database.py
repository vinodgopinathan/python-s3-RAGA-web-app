#!/usr/bin/env python3
"""
Manual Database Initialization Script
Run this script to initialize or update the database schema when needed.
"""

import os
import sys
import logging
from pathlib import Path

# Add the backend source directory to Python path
backend_path = Path(__file__).parent / "backend" / "src"
sys.path.insert(0, str(backend_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_database():
    """Initialize the database schema"""
    try:
        from utils.enhanced_vector_db_helper import EnhancedVectorDBHelper
        
        # Force database initialization even if SKIP_DB_INIT is set
        original_skip = os.environ.get('SKIP_DB_INIT')
        os.environ['SKIP_DB_INIT'] = 'false'
        
        # Create database helper (this will initialize schema)
        db_helper = EnhancedVectorDBHelper()
        
        # Force schema creation
        db_helper._create_schema()
        
        # Restore original setting
        if original_skip is not None:
            os.environ['SKIP_DB_INIT'] = original_skip
        
        logger.info("Database initialization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False

def check_database():
    """Check if database schema exists"""
    try:
        from utils.enhanced_vector_db_helper import EnhancedVectorDBHelper
        
        # Temporarily skip auto-initialization
        original_skip = os.environ.get('SKIP_DB_INIT')
        os.environ['SKIP_DB_INIT'] = 'true'
        
        db_helper = EnhancedVectorDBHelper()
        
        # Check if tables exist
        conn = db_helper.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('documents', 'document_chunks')
        """)
        
        table_count = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        # Restore original setting
        if original_skip is not None:
            os.environ['SKIP_DB_INIT'] = original_skip
        
        if table_count >= 2:
            logger.info("Database schema exists and is ready")
            return True
        else:
            logger.warning("Database schema is missing or incomplete")
            return False
            
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return False

def show_help():
    print("Database Initialization Script")
    print("")
    print("Usage: python init_database.py [COMMAND]")
    print("")
    print("Commands:")
    print("  init    - Initialize database schema (default)")
    print("  check   - Check if database schema exists")
    print("  help    - Show this help message")
    print("")
    print("Environment Variables:")
    print("  SKIP_DB_INIT=true   - Skip automatic database initialization")

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "init"
    
    if command == "init":
        success = init_database()
        sys.exit(0 if success else 1)
    elif command == "check":
        exists = check_database()
        sys.exit(0 if exists else 1)
    elif command == "help":
        show_help()
        sys.exit(0)
    else:
        print(f"Unknown command: {command}")
        show_help()
        sys.exit(1)
