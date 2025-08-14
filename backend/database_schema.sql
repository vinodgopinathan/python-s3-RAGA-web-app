-- PostgreSQL Database Schema for Advanced RAG Chunking System
-- This creates the necessary tables and indexes for storing document chunks with vector embeddings
-- Uses conditional creation to avoid conflicts on repeated runs

-- Enable pgvector extension for vector operations (only if not exists)
CREATE EXTENSION IF NOT EXISTS vector;

-- Main documents table to store document metadata
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path TEXT NOT NULL,
    s3_key TEXT,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size BIGINT,
    content_type TEXT,
    source_type TEXT NOT NULL CHECK (source_type IN ('local', 's3')),
    chunking_method TEXT NOT NULL CHECK (chunking_method IN ('recursive', 'semantic', 'agentic', 'adaptive', 'sentence')),
    chunk_size INTEGER DEFAULT 1000,
    chunk_overlap INTEGER DEFAULT 200,
    total_chunks INTEGER DEFAULT 0,
    processing_status TEXT DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Processing errors table to track all errors during document processing
CREATE TABLE IF NOT EXISTS processing_errors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    file_name TEXT NOT NULL,
    file_path TEXT,
    s3_key TEXT,
    failure_scope TEXT NOT NULL DEFAULT 'document' CHECK (failure_scope IN ('document', 'chunk')), -- NEW: Distinguish document vs chunk failures
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

-- Document chunks table with vector embeddings
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_length INTEGER NOT NULL,
    embedding vector(384), -- 384 dimensions for all-MiniLM-L6-v2 model
    
    -- Chunking metadata
    chunking_method TEXT NOT NULL,
    separator_used TEXT,
    char_count INTEGER,
    sentence_count INTEGER,
    
    -- Semantic chunking specific metadata
    semantic_boundary BOOLEAN DEFAULT FALSE,
    similarity_score FLOAT,
    
    -- Agentic chunking specific metadata
    llm_reasoning TEXT,
    llm_confidence FLOAT,
    llm_provider TEXT,
    
    -- Adaptive chunking metadata
    adaptive_method TEXT,
    text_analysis JSONB,
    chosen_strategy TEXT,
    
    -- General metadata
    metadata JSONB DEFAULT '{}',
    source_type TEXT NOT NULL,
    s3_key TEXT,
    file_path TEXT,
    content_type TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    UNIQUE(document_id, chunk_index)
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_chunking_method ON documents(chunking_method);
CREATE INDEX IF NOT EXISTS idx_documents_processing_status ON documents(processing_status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);

-- Processing errors indexes
CREATE INDEX IF NOT EXISTS idx_processing_errors_document_id ON processing_errors(document_id);
CREATE INDEX IF NOT EXISTS idx_processing_errors_failure_scope ON processing_errors(failure_scope);
CREATE INDEX IF NOT EXISTS idx_processing_errors_error_type ON processing_errors(error_type);
CREATE INDEX IF NOT EXISTS idx_processing_errors_processing_stage ON processing_errors(processing_stage);
CREATE INDEX IF NOT EXISTS idx_processing_errors_resolved ON processing_errors(resolved);
CREATE INDEX IF NOT EXISTS idx_processing_errors_created_at ON processing_errors(created_at);
CREATE INDEX IF NOT EXISTS idx_processing_errors_file_name ON processing_errors(file_name);
CREATE INDEX IF NOT EXISTS idx_processing_errors_scope_type ON processing_errors(failure_scope, error_type);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunking_method ON document_chunks(chunking_method);
CREATE INDEX IF NOT EXISTS idx_chunks_source_type ON document_chunks(source_type);
CREATE INDEX IF NOT EXISTS idx_chunks_content_length ON document_chunks(content_length);
CREATE INDEX IF NOT EXISTS idx_chunks_created_at ON document_chunks(created_at);

-- Vector similarity search index (HNSW for better performance)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine ON document_chunks 
USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- GIN index for JSONB metadata queries
CREATE INDEX IF NOT EXISTS idx_chunks_metadata_gin ON document_chunks USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_chunks_text_analysis_gin ON document_chunks USING gin(text_analysis);

-- Full-text search index on content
CREATE INDEX IF NOT EXISTS idx_chunks_content_fts ON document_chunks USING gin(to_tsvector('english', content));

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_chunks_updated_at ON document_chunks;
CREATE TRIGGER update_chunks_updated_at 
    BEFORE UPDATE ON document_chunks 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_processing_errors_updated_at ON processing_errors;
CREATE TRIGGER update_processing_errors_updated_at 
    BEFORE UPDATE ON processing_errors 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Useful views for analytics and monitoring
CREATE OR REPLACE VIEW chunking_analytics AS
SELECT 
    chunking_method,
    source_type,
    COUNT(*) as total_documents,
    SUM(total_chunks) as total_chunks,
    AVG(total_chunks) as avg_chunks_per_doc,
    AVG(chunk_size) as avg_chunk_size,
    AVG(chunk_overlap) as avg_chunk_overlap,
    COUNT(CASE WHEN processing_status = 'completed' THEN 1 END) as completed_docs,
    COUNT(CASE WHEN processing_status = 'failed' THEN 1 END) as failed_docs
FROM documents 
GROUP BY chunking_method, source_type
ORDER BY total_documents DESC;

CREATE OR REPLACE VIEW chunk_quality_metrics AS
SELECT 
    chunking_method,
    COUNT(*) as total_chunks,
    AVG(content_length) as avg_content_length,
    MIN(content_length) as min_content_length,
    MAX(content_length) as max_content_length,
    AVG(CASE WHEN sentence_count IS NOT NULL THEN sentence_count END) as avg_sentences,
    AVG(CASE WHEN similarity_score IS NOT NULL THEN similarity_score END) as avg_similarity,
    AVG(CASE WHEN llm_confidence IS NOT NULL THEN llm_confidence END) as avg_llm_confidence,
    COUNT(CASE WHEN semantic_boundary = true THEN 1 END) as semantic_boundaries_count
FROM document_chunks 
GROUP BY chunking_method
ORDER BY total_chunks DESC;

-- Error analytics view
CREATE OR REPLACE VIEW error_analytics AS
SELECT 
    failure_scope,
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
GROUP BY failure_scope, error_type, processing_stage
ORDER BY failure_scope, error_count DESC;

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

-- Sample queries for testing
/*
-- Query to find similar chunks using vector search
SELECT 
    dc.content,
    dc.chunking_method,
    dc.similarity_score,
    d.file_name,
    dc.embedding <=> '[vector_here]'::vector AS distance
FROM document_chunks dc
JOIN documents d ON dc.document_id = d.id
ORDER BY dc.embedding <=> '[vector_here]'::vector
LIMIT 10;

-- Query to analyze chunking performance by method
SELECT * FROM chunking_analytics;

-- Query to see chunk quality metrics
SELECT * FROM chunk_quality_metrics;

-- Query to find documents by chunking method and status
SELECT d.*, COUNT(dc.id) as actual_chunks
FROM documents d
LEFT JOIN document_chunks dc ON d.id = dc.document_id
WHERE d.chunking_method = 'semantic' AND d.processing_status = 'completed'
GROUP BY d.id
ORDER BY d.created_at DESC;
*/
