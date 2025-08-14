-- SQL Queries to Distinguish Document vs Chunk Failures
-- After implementing the failure_scope field

-- 1. Summary of all failures by scope
SELECT 
    failure_scope,
    COUNT(*) as total_errors,
    COUNT(DISTINCT document_id) as affected_documents,
    COUNT(CASE WHEN resolved = true THEN 1 END) as resolved_errors,
    COUNT(CASE WHEN resolved = false THEN 1 END) as unresolved_errors
FROM processing_errors 
GROUP BY failure_scope
ORDER BY total_errors DESC;

-- 2. Chunk-specific failures with details
SELECT 
    pe.file_name,
    pe.error_type,
    pe.error_message,
    pe.error_details->>'failed_chunk_count' as failed_chunks,
    pe.error_details->>'successfully_stored_chunks' as stored_chunks,
    pe.error_details->>'total_chunks_in_document' as total_chunks,
    pe.created_at
FROM processing_errors pe
WHERE pe.failure_scope = 'chunk'
ORDER BY pe.created_at DESC
LIMIT 10;

-- 3. Document-level failures (entire document processing failed)
SELECT 
    pe.file_name,
    pe.error_type,
    pe.error_message,
    pe.processing_stage,
    pe.retry_count,
    pe.is_recoverable,
    pe.created_at
FROM processing_errors pe
WHERE pe.failure_scope = 'document'
ORDER BY pe.created_at DESC
LIMIT 10;

-- 4. Documents with both document-level AND chunk-level failures
SELECT 
    d.file_name,
    COUNT(CASE WHEN pe.failure_scope = 'document' THEN 1 END) as document_errors,
    COUNT(CASE WHEN pe.failure_scope = 'chunk' THEN 1 END) as chunk_errors,
    STRING_AGG(DISTINCT pe.error_type, ', ') as error_types,
    MAX(pe.created_at) as latest_error
FROM documents d
JOIN processing_errors pe ON d.id = pe.document_id
GROUP BY d.id, d.file_name
HAVING COUNT(CASE WHEN pe.failure_scope = 'document' THEN 1 END) > 0 
   AND COUNT(CASE WHEN pe.failure_scope = 'chunk' THEN 1 END) > 0
ORDER BY latest_error DESC;

-- 5. Error type breakdown by scope
SELECT 
    failure_scope,
    error_type,
    processing_stage,
    COUNT(*) as error_count,
    COUNT(DISTINCT document_id) as affected_documents,
    ROUND(AVG(retry_count), 2) as avg_retries
FROM processing_errors 
GROUP BY failure_scope, error_type, processing_stage
ORDER BY failure_scope, error_count DESC;

-- 6. Find documents where chunks failed but document was marked as successful
SELECT 
    d.file_name,
    d.processing_status,
    d.total_chunks,
    COUNT(dc.id) as actual_stored_chunks,
    SUM((pe.error_details->>'failed_chunk_count')::int) as failed_chunks,
    STRING_AGG(pe.error_type, ', ') as chunk_error_types
FROM documents d
LEFT JOIN document_chunks dc ON d.id = dc.document_id
LEFT JOIN processing_errors pe ON d.id = pe.document_id AND pe.failure_scope = 'chunk'
WHERE d.processing_status = 'completed'
GROUP BY d.id, d.file_name, d.processing_status, d.total_chunks
HAVING SUM((pe.error_details->>'failed_chunk_count')::int) > 0
ORDER BY failed_chunks DESC;

-- 7. Performance comparison: documents with only chunk failures vs mixed failures
WITH failure_stats AS (
    SELECT 
        document_id,
        COUNT(CASE WHEN failure_scope = 'document' THEN 1 END) as doc_failures,
        COUNT(CASE WHEN failure_scope = 'chunk' THEN 1 END) as chunk_failures,
        SUM((error_details->>'failed_chunk_count')::int) as total_failed_chunks
    FROM processing_errors
    GROUP BY document_id
)
SELECT 
    CASE 
        WHEN fs.doc_failures > 0 AND fs.chunk_failures > 0 THEN 'Mixed Failures'
        WHEN fs.doc_failures > 0 AND fs.chunk_failures = 0 THEN 'Document Only'
        WHEN fs.doc_failures = 0 AND fs.chunk_failures > 0 THEN 'Chunks Only'
        ELSE 'No Failures'
    END as failure_pattern,
    COUNT(*) as document_count,
    AVG(d.total_chunks) as avg_total_chunks,
    AVG(COALESCE(fs.total_failed_chunks, 0)) as avg_failed_chunks,
    AVG(CASE WHEN d.total_chunks > 0 THEN 
        (d.total_chunks - COALESCE(fs.total_failed_chunks, 0))::float / d.total_chunks * 100 
        ELSE 100 END) as avg_success_rate_percent
FROM documents d
LEFT JOIN failure_stats fs ON d.id = fs.document_id
GROUP BY 
    CASE 
        WHEN fs.doc_failures > 0 AND fs.chunk_failures > 0 THEN 'Mixed Failures'
        WHEN fs.doc_failures > 0 AND fs.chunk_failures = 0 THEN 'Document Only'
        WHEN fs.doc_failures = 0 AND fs.chunk_failures > 0 THEN 'Chunks Only'
        ELSE 'No Failures'
    END
ORDER BY document_count DESC;
