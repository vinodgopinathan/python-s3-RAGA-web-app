#!/bin/bash

# AWS LLM RAGA Project - External Database Setup Script
# This script sets up your external PostgreSQL database with the required schema and extensions

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if required tools are available
check_requirements() {
    if ! command -v psql &> /dev/null; then
        print_error "psql (PostgreSQL client) not found. Please install PostgreSQL client tools."
        exit 1
    fi
    print_success "PostgreSQL client is available"
}

# Function to load environment variables
load_env() {
    if [ -f ".env" ]; then
        print_info "Loading environment from .env file"
        export $(cat .env | grep -v '^#' | xargs)
    elif [ -f ".env.external" ]; then
        print_info "Loading environment from .env.external file"
        export $(cat .env.external | grep -v '^#' | xargs)
    else
        print_error "No environment file found. Please create .env or .env.external"
        exit 1
    fi
}

# Function to test database connection
test_connection() {
    print_info "Testing database connection..."
    
    local db_host="${DB_HOST:-${POSTGRES_HOST}}"
    local db_port="${DB_PORT:-${POSTGRES_PORT:-5432}}"
    local db_user="${DB_USER:-${POSTGRES_USER:-postgres}}"
    local db_name="${DB_NAME:-${POSTGRES_DB:-rag_system}}"
    
    if [ -z "$db_host" ] || [ -z "$DB_PASSWORD" ] && [ -z "$POSTGRES_PASSWORD" ]; then
        print_error "Missing database configuration. Please check your .env file."
        print_info "Required variables: DB_HOST (or POSTGRES_HOST), DB_PASSWORD (or POSTGRES_PASSWORD)"
        exit 1
    fi
    
    local password="${DB_PASSWORD:-${POSTGRES_PASSWORD}}"
    
    export PGPASSWORD="$password"
    
    if psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -c "SELECT 1;" > /dev/null 2>&1; then
        print_success "Database connection successful"
        return 0
    else
        print_error "Failed to connect to database"
        print_info "Connection details:"
        print_info "  Host: $db_host"
        print_info "  Port: $db_port"
        print_info "  User: $db_user"
        print_info "  Database: $db_name"
        return 1
    fi
}

# Function to create database if it doesn't exist
create_database() {
    print_info "Checking if database exists..."
    
    local db_host="${DB_HOST:-${POSTGRES_HOST}}"
    local db_port="${DB_PORT:-${POSTGRES_PORT:-5432}}"
    local db_user="${DB_USER:-${POSTGRES_USER:-postgres}}"
    local db_name="${DB_NAME:-${POSTGRES_DB:-rag_system}}"
    local password="${DB_PASSWORD:-${POSTGRES_PASSWORD}}"
    
    export PGPASSWORD="$password"
    
    # Check if database exists
    if psql -h "$db_host" -p "$db_port" -U "$db_user" -d postgres -tc "SELECT 1 FROM pg_database WHERE datname = '$db_name'" | grep -q 1; then
        print_success "Database '$db_name' already exists"
    else
        print_info "Creating database '$db_name'..."
        psql -h "$db_host" -p "$db_port" -U "$db_user" -d postgres -c "CREATE DATABASE $db_name;"
        print_success "Database '$db_name' created successfully"
    fi
}

# Function to install pgvector extension
install_pgvector() {
    print_info "Installing pgvector extension..."
    
    local db_host="${DB_HOST:-${POSTGRES_HOST}}"
    local db_port="${DB_PORT:-${POSTGRES_PORT:-5432}}"
    local db_user="${DB_USER:-${POSTGRES_USER:-postgres}}"
    local db_name="${DB_NAME:-${POSTGRES_DB:-rag_system}}"
    local password="${DB_PASSWORD:-${POSTGRES_PASSWORD}}"
    
    export PGPASSWORD="$password"
    
    # Check if pgvector is already installed
    if psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -tc "SELECT 1 FROM pg_extension WHERE extname = 'vector'" | grep -q 1; then
        print_success "pgvector extension already installed"
    else
        print_info "Installing pgvector extension..."
        if psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -c "CREATE EXTENSION IF NOT EXISTS vector;" > /dev/null 2>&1; then
            print_success "pgvector extension installed successfully"
        else
            print_error "Failed to install pgvector extension"
            print_warning "Make sure your PostgreSQL instance supports pgvector"
            print_info "For AWS RDS, you may need to:"
            print_info "1. Use PostgreSQL 11+ with pgvector support"
            print_info "2. Add 'pgvector' to shared_preload_libraries parameter"
            print_info "3. Restart the RDS instance"
            return 1
        fi
    fi
}

# Function to run database schema
setup_schema() {
    print_info "Setting up database schema..."
    
    local db_host="${DB_HOST:-${POSTGRES_HOST}}"
    local db_port="${DB_PORT:-${POSTGRES_PORT:-5432}}"
    local db_user="${DB_USER:-${POSTGRES_USER:-postgres}}"
    local db_name="${DB_NAME:-${POSTGRES_DB:-rag_system}}"
    local password="${DB_PASSWORD:-${POSTGRES_PASSWORD}}"
    
    export PGPASSWORD="$password"
    
    # Check if tables already exist
    local existing_tables=$(psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -tc "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('documents', 'document_chunks');")
    
    if [ "$existing_tables" = "2" ]; then
        print_warning "Database schema already exists. Skipping schema creation."
        print_info "Use 'force-schema' command to recreate schema."
        return 0
    fi
    
    if [ -f "backend/database_schema.sql" ]; then
        psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -f backend/database_schema.sql
        print_success "Database schema applied successfully"
    else
        print_error "Database schema file not found: backend/database_schema.sql"
        return 1
    fi
}

# Function to force recreate schema
force_schema() {
    print_warning "Force recreating database schema..."
    
    local db_host="${DB_HOST:-${POSTGRES_HOST}}"
    local db_port="${DB_PORT:-${POSTGRES_PORT:-5432}}"
    local db_user="${DB_USER:-${POSTGRES_USER:-postgres}}"
    local db_name="${DB_NAME:-${POSTGRES_DB:-rag_system}}"
    local password="${DB_PASSWORD:-${POSTGRES_PASSWORD}}"
    
    export PGPASSWORD="$password"
    
    # Drop existing tables if they exist
    psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -c "
        DROP TABLE IF EXISTS document_chunks CASCADE;
        DROP TABLE IF EXISTS documents CASCADE;
        DROP TABLE IF EXISTS processing_jobs CASCADE;
        DROP VIEW IF EXISTS chunking_analytics CASCADE;
        DROP VIEW IF EXISTS chunk_quality_metrics CASCADE;
        DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;
    "
    
    # Apply schema
    if [ -f "backend/database_schema.sql" ]; then
        psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -f backend/database_schema.sql
        print_success "Database schema recreated successfully"
    else
        print_error "Database schema file not found: backend/database_schema.sql"
        return 1
    fi
}

# Function to verify setup
verify_setup() {
    print_info "Verifying database setup..."
    
    local db_host="${DB_HOST:-${POSTGRES_HOST}}"
    local db_port="${DB_PORT:-${POSTGRES_PORT:-5432}}"
    local db_user="${DB_USER:-${POSTGRES_USER:-postgres}}"
    local db_name="${DB_NAME:-${POSTGRES_DB:-rag_system}}"
    local password="${DB_PASSWORD:-${POSTGRES_PASSWORD}}"
    
    export PGPASSWORD="$password"
    
    # Check if tables exist
    local tables=$(psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -tc "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name IN ('documents', 'document_chunks', 'processing_jobs');")
    
    if [ "$tables" = "3" ]; then
        print_success "All required tables are present"
    else
        print_warning "Some tables may be missing. Found $tables out of 3 expected tables."
    fi
    
    # Check if pgvector is working
    if psql -h "$db_host" -p "$db_port" -U "$db_user" -d "$db_name" -tc "SELECT vector_dims(ARRAY[1,2,3]::vector);" 2>/dev/null | grep -q 3; then
        print_success "pgvector extension is working correctly"
    else
        print_warning "pgvector extension may not be working properly"
    fi
}

# Function to show connection info
show_connection_info() {
    print_info "Database connection information:"
    echo "  Host: ${DB_HOST:-${POSTGRES_HOST}}"
    echo "  Port: ${DB_PORT:-${POSTGRES_PORT:-5432}}"
    echo "  Database: ${DB_NAME:-${POSTGRES_DB:-rag_system}}"
    echo "  User: ${DB_USER:-${POSTGRES_USER:-postgres}}"
    echo ""
    print_info "To connect manually:"
    echo "  psql -h ${DB_HOST:-${POSTGRES_HOST}} -p ${DB_PORT:-${POSTGRES_PORT:-5432}} -U ${DB_USER:-${POSTGRES_USER:-postgres}} -d ${DB_NAME:-${POSTGRES_DB:-rag_system}}"
}

# Function to show help
show_help() {
    echo "AWS LLM RAGA Project - External Database Setup Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  setup        - Complete database setup (creates schema only if needed)"
    echo "  test         - Test database connection"
    echo "  schema       - Apply database schema only (skips if exists)"
    echo "  force-schema - Force recreate database schema (drops and recreates)"
    echo "  verify       - Verify database setup"
    echo "  info         - Show connection information"
    echo "  help         - Show this help message"
    echo ""
    echo "Prerequisites:"
    echo "1. PostgreSQL client (psql) installed"
    echo "2. .env or .env.external file with database configuration"
    echo "3. Network access to your PostgreSQL database"
    echo "4. Database user with CREATE DATABASE and CREATE EXTENSION privileges"
}

# Main script logic
case "${1:-setup}" in
    "setup")
        check_requirements
        load_env
        test_connection || exit 1
        create_database
        install_pgvector || exit 1
        setup_schema || exit 1
        verify_setup
        show_connection_info
        print_success "Database setup completed successfully!"
        ;;
    "test")
        check_requirements
        load_env
        test_connection && print_success "Connection test passed!" || exit 1
        ;;
    "schema")
        check_requirements
        load_env
        test_connection || exit 1
        setup_schema || exit 1
        print_success "Schema applied successfully!"
        ;;
    "force-schema")
        check_requirements
        load_env
        test_connection || exit 1
        force_schema || exit 1
        verify_setup
        print_success "Database schema recreated successfully!"
        ;;
    "verify")
        check_requirements
        load_env
        test_connection || exit 1
        verify_setup
        ;;
    "info")
        load_env
        show_connection_info
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
