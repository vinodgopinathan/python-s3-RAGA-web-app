import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ChunkingInterface.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

const ChunkingInterface = () => {
  // Directory Processing State
  const [directoryPath, setDirectoryPath] = useState('/');
  const [directoryChunkingMethod, setDirectoryChunkingMethod] = useState('recursive');
  const [directoryChunkSize, setDirectoryChunkSize] = useState(1000);
  const [directoryChunkOverlap, setDirectoryChunkOverlap] = useState(200);
  const [recursive, setRecursive] = useState(true);
  const [directoryProcessing, setDirectoryProcessing] = useState(false);
  const [currentJob, setCurrentJob] = useState(null);

  // S3 Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [s3Path, setS3Path] = useState('');
  const [s3ChunkingMethod, setS3ChunkingMethod] = useState('recursive');
  const [s3ChunkSize, setS3ChunkSize] = useState(1000);
  const [s3ChunkOverlap, setS3ChunkOverlap] = useState(200);
  const [s3Processing, setS3Processing] = useState(false);
  const [s3Result, setS3Result] = useState(null);
  const [s3ProgressStage, setS3ProgressStage] = useState('');
  const [s3ProgressPercent, setS3ProgressPercent] = useState(0);

  // General State
  const [chunkingMethods, setChunkingMethods] = useState([]);
  const [supportedExtensions, setSupportedExtensions] = useState([]);
  const [stats, setStats] = useState(null);
  const [recentDocuments, setRecentDocuments] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState(null);

  // Load initial data
  useEffect(() => {
    loadChunkingMethods();
    loadSupportedExtensions();
    loadStats();
    loadRecentDocuments();
  }, []);

  // Poll job status
  useEffect(() => {
    let interval = null;
    if (currentJob && (currentJob.status === 'processing' || currentJob.status === 'started')) {
      interval = setInterval(() => {
        checkJobStatus(currentJob.job_id);
      }, 1000); // Poll every 1 second for better responsiveness
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [currentJob]);

  const loadChunkingMethods = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/chunking-methods`);
      setChunkingMethods(response.data.methods);
    } catch (error) {
      console.error('Error loading chunking methods:', error);
    }
  };

  const loadSupportedExtensions = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/supported-extensions`);
      setSupportedExtensions(response.data.extensions);
    } catch (error) {
      console.error('Error loading supported extensions:', error);
    }
  };

  const loadStats = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/stats`);
      setStats(response.data.stats);
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  const loadRecentDocuments = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/recent-documents?limit=10`);
      setRecentDocuments(response.data.documents);
    } catch (error) {
      console.error('Error loading recent documents:', error);
    }
  };

  const processDirectory = async () => {
    if (!directoryPath.trim()) {
      alert('Please enter a directory path');
      return;
    }

    setDirectoryProcessing(true);
    setCurrentJob(null); // Clear previous job
    
    try {
      console.log('DEBUG: About to send request with these values:');
      console.log('  directoryPath:', directoryPath);
      console.log('  directoryChunkingMethod:', directoryChunkingMethod);
      
      // Use environment variable to prevent minification issues
      const payload = {
        directory_path: directoryPath,
        chunking_method: directoryChunkingMethod,
        chunk_size: directoryChunkSize,
        chunk_overlap: directoryChunkOverlap,
        recursive: recursive
        // No source_type needed - backend always uses S3
      };
      
      console.log('=== FINAL DEBUG: Request payload ===', payload);
      console.log('Request keys:', Object.keys(payload));
      
      const response = await fetch(`${API_BASE_URL}/process-directory`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const responseData = await response.json();
      console.log('=== FINAL DEBUG: Response ===', responseData);

      if (responseData.status === 'started') {
        setCurrentJob({
          job_id: responseData.job_id,
          status: 'started',
          message: responseData.message,
          files_found: responseData.files_found,
          total_files: responseData.files_found,
          processed_files: 0,
          failed_files: 0,
          progress_percentage: 0,
          current_file: '',
          start_time: new Date().toISOString()
        });
      } else {
        alert(responseData.message || 'Processing completed immediately');
        setDirectoryProcessing(false);
      }
    } catch (error) {
      console.error('Error processing directory:', error);
      alert('Error processing directory: ' + error.message);
      setDirectoryProcessing(false);
      setCurrentJob(null);
    }
  };

  const checkJobStatus = async (jobId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/job-status/${jobId}`);
      const jobStatus = response.data.job_status;
      
      // Debug logging for progress tracking
      console.log(`Job ${jobId} Status:`, {
        status: jobStatus.status,
        processed_files: jobStatus.processed_files,
        total_files: jobStatus.total_files,
        progress_percentage: jobStatus.progress_percentage,
        current_file: jobStatus.current_file
      });
      
      setCurrentJob(jobStatus);
      
      if (jobStatus.status === 'completed' || jobStatus.status === 'failed' || jobStatus.status === 'cancelled') {
        setDirectoryProcessing(false);
        loadStats();
        loadRecentDocuments();
        
        // Show completion message
        if (jobStatus.status === 'completed') {
          alert(`Processing completed! ${jobStatus.processed_files} files processed successfully.`);
        } else if (jobStatus.status === 'failed') {
          alert(`Processing failed: ${jobStatus.error || 'Unknown error'}`);
        } else if (jobStatus.status === 'cancelled') {
          alert('Processing was cancelled.');
        }
      }
    } catch (error) {
      console.error('Error checking job status:', error);
    }
  };

  const uploadToS3 = async () => {
    if (!selectedFile) {
      alert('Please select a file');
      return;
    }

    if (!s3Path.trim()) {
      alert('Please enter an S3 path');
      return;
    }

    setS3Processing(true);
    setS3Result(null);
    setS3ProgressPercent(0);

    try {
      // Stage 1: Preparing upload
      setS3ProgressStage('Preparing file upload...');
      setS3ProgressPercent(10);
      await new Promise(resolve => setTimeout(resolve, 500)); // Small delay for UX

      // Stage 2: Uploading to S3
      setS3ProgressStage('Uploading to S3...');
      setS3ProgressPercent(30);
      
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('s3_path', s3Path);
      formData.append('chunking_method', s3ChunkingMethod);
      formData.append('chunk_size', s3ChunkSize);
      formData.append('chunk_overlap', s3ChunkOverlap);

      // Stage 3: Processing document
      setS3ProgressStage('Processing document...');
      setS3ProgressPercent(60);

      const response = await axios.post(`${API_BASE_URL}/upload-s3`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          // Calculate upload progress (30% to 90% of total progress)
          const uploadPercent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          const totalProgress = 30 + (uploadPercent * 0.6); // Scale to 30-90%
          setS3ProgressPercent(Math.min(totalProgress, 90));
        }
      });

      // Stage 4: Finalizing
      setS3ProgressStage('Finalizing...');
      setS3ProgressPercent(95);
      await new Promise(resolve => setTimeout(resolve, 300)); // Small delay for UX

      // Stage 5: Complete
      setS3ProgressStage('Complete!');
      setS3ProgressPercent(100);

      setS3Result(response.data);
      if (response.data.status === 'success') {
        loadStats();
        loadRecentDocuments();
        setSelectedFile(null);
        setS3Path('');
        
        // Show completion message briefly
        setTimeout(() => {
          setS3ProgressStage('');
          setS3ProgressPercent(0);
        }, 1500);
      }
    } catch (error) {
      console.error('Error uploading to S3:', error);
      setS3ProgressStage('Error occurred');
      setS3ProgressPercent(100);
      setS3Result({
        status: 'error',
        message: error.response?.data?.message || error.message
      });
    } finally {
      setS3Processing(false);
    }
  };

  const searchDocuments = async () => {
    if (!searchQuery.trim()) {
      alert('Please enter a search query');
      return;
    }

    try {
      const response = await axios.post(`${API_BASE_URL}/rag/hybrid-search`, {
        query: searchQuery,
        k: 10
      });
      setSearchResults(response.data);
    } catch (error) {
      console.error('Error searching documents:', error);
      alert('Search failed: ' + (error.response?.data?.message || error.message));
    }
  };

  const cancelJob = async () => {
    if (currentJob && currentJob.job_id) {
      try {
        await axios.post(`${API_BASE_URL}/cancel-job/${currentJob.job_id}`);
        setCurrentJob({ ...currentJob, status: 'cancelled' });
        setDirectoryProcessing(false);
      } catch (error) {
        console.error('Error cancelling job:', error);
      }
    }
  };

  const deleteDocument = async (documentId) => {
    if (window.confirm('Are you sure you want to delete this document and all its chunks?')) {
      try {
        await axios.delete(`${API_BASE_URL}/delete-document/${documentId}`);
        loadRecentDocuments();
        loadStats();
        alert('Document deleted successfully');
      } catch (error) {
        console.error('Error deleting document:', error);
        alert('Error deleting document: ' + (error.response?.data?.message || error.message));
      }
    }
  };

  return (
    <div className="chunking-interface">
      <header className="header">
        <h1>🧠 Advanced Document Chunking System</h1>
        <p>Process files with semantic, agentic, and adaptive chunking methods</p>
      </header>

      <div className="container">
        {/* Statistics Dashboard */}
        <div className="stats-section">
          <div className="stats-header">
            <h2>📊 System Statistics</h2>
            <button 
              onClick={loadStats} 
              className="btn btn-secondary"
              title="Refresh statistics"
            >
              🔄 Refresh Stats
            </button>
          </div>
          {stats ? (
            <div className="stats-grid">
              <div className="stat-card">
                <h3>Total Documents</h3>
                <div className="stat-value">{stats.overall.total_documents}</div>
              </div>
              <div className="stat-card">
                <h3>Total Chunks</h3>
                <div className="stat-value">{stats.overall.total_chunks}</div>
              </div>
              <div className="stat-card">
                <h3>Avg Chunk Length</h3>
                <div className="stat-value">{Math.round(stats.overall.avg_chunk_length || 0)}</div>
              </div>
              <div className="stat-card">
                <h3>Completed</h3>
                <div className="stat-value">{stats.overall.completed_docs}</div>
              </div>
              <div className="stat-card">
                <h3>Processing</h3>
                <div className="stat-value">{stats.overall.processing_docs}</div>
              </div>
              <div className="stat-card">
                <h3>Failed</h3>
                <div className="stat-value">{stats.overall.failed_docs}</div>
              </div>
            </div>
          ) : (
            <div className="stats-loading">
              <p>Loading statistics... <button onClick={loadStats} className="btn btn-link">Retry</button></p>
            </div>
          )}
        </div>

        <div className="main-sections">
          {/* Section 1: S3 Directory Processing */}
          <div className="section">
            <h2>📁 S3 Directory Processing</h2>
            <div className="form-group">
              <label>S3 Path (leave as "/" to process all files):</label>
              <input
                type="text"
                value={directoryPath}
                onChange={(e) => setDirectoryPath(e.target.value)}
                placeholder="/ (root) or specific/folder/path"
                disabled={directoryProcessing}
              />
              <small className="form-help">
                Enter "/" to process all files in the S3 bucket, or specify a folder path like "documents/" to process files in that folder.
              </small>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Chunking Method:</label>
                <select
                  value={directoryChunkingMethod}
                  onChange={(e) => setDirectoryChunkingMethod(e.target.value)}
                  disabled={directoryProcessing}
                >
                  {chunkingMethods.map(method => (
                    <option key={method.name} value={method.name}>
                      {method.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Chunk Size:</label>
                <input
                  type="number"
                  value={directoryChunkSize}
                  onChange={(e) => setDirectoryChunkSize(parseInt(e.target.value))}
                  min="100"
                  max="4000"
                  disabled={directoryProcessing}
                />
              </div>

              <div className="form-group">
                <label>Chunk Overlap:</label>
                <input
                  type="number"
                  value={directoryChunkOverlap}
                  onChange={(e) => setDirectoryChunkOverlap(parseInt(e.target.value))}
                  min="0"
                  max="1000"
                  disabled={directoryProcessing}
                />
              </div>
            </div>

            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  checked={recursive}
                  onChange={(e) => setRecursive(e.target.checked)}
                  disabled={directoryProcessing}
                />
                Process subdirectories recursively
              </label>
            </div>

            <button
              onClick={processDirectory}
              disabled={directoryProcessing}
              className="btn btn-primary"
            >
              {directoryProcessing ? 'Processing...' : 'Process Directory'}
            </button>

            {/* Enhanced Job Status with Progress Bar */}
            {currentJob && (
              <div className="job-status">
                <h3>🔄 Processing Status</h3>
                
                {/* Progress Bar */}
                <div className="progress-container">
                  <div className="progress-header">
                    <span className="progress-text">
                      {currentJob.status === 'completed' ? '✅ Completed' : 
                       currentJob.status === 'failed' ? '❌ Failed' :
                       currentJob.status === 'cancelled' ? '🚫 Cancelled' :
                       `Processing... ${currentJob.processed_files || 0}/${currentJob.total_files || 0}`}
                    </span>
                    <span className="progress-percentage">
                      {currentJob.progress_percentage || 0}%
                    </span>
                  </div>
                  
                  <div className="progress-bar">
                    <div 
                      className={`progress-fill ${
                        currentJob.status === 'completed' ? 'completed' :
                        currentJob.status === 'failed' ? 'failed' :
                        currentJob.status === 'cancelled' ? 'cancelled' : 'processing'
                      }`}
                      style={{ 
                        width: `${currentJob.progress_percentage || 0}%`,
                        transition: 'width 0.8s ease-out', // Slower transition to see updates
                        minWidth: currentJob.progress_percentage > 0 ? '10px' : '0px' // Ensure visibility
                      }}
                      title={`${currentJob.progress_percentage || 0}% complete`}
                    ></div>
                  </div>
                </div>

                {/* Job Details */}
                <div className="job-details">
                  <div className="detail-row">
                    <span className="detail-label">Job ID:</span>
                    <span className="detail-value">{currentJob.job_id}</span>
                  </div>
                  
                  <div className="detail-row">
                    <span className="detail-label">Status:</span>
                    <span className={`detail-value status-${currentJob.status}`}>
                      {currentJob.status.toUpperCase()}
                    </span>
                  </div>
                  
                  {currentJob.total_files > 0 && (
                    <div className="detail-row">
                      <span className="detail-label">Progress:</span>
                      <span className="detail-value">
                        {currentJob.processed_files || 0} of {currentJob.total_files} files
                        {currentJob.failed_files > 0 && (
                          <span className="failed-count"> ({currentJob.failed_files} failed)</span>
                        )}
                      </span>
                    </div>
                  )}
                  
                  {currentJob.current_file && (
                    <div className="detail-row">
                      <span className="detail-label">Current File:</span>
                      <span className="detail-value current-file">
                        {currentJob.current_file.split('/').pop()}
                      </span>
                    </div>
                  )}
                  
                  {currentJob.duration && (
                    <div className="detail-row">
                      <span className="detail-label">Duration:</span>
                      <span className="detail-value">
                        {currentJob.duration.toFixed(1)}s
                      </span>
                    </div>
                  )}
                </div>

                {/* Action Buttons */}
                <div className="job-actions">
                  {(currentJob.status === 'processing' || currentJob.status === 'started') && (
                    <button onClick={cancelJob} className="btn btn-danger">
                      🚫 Cancel Job
                    </button>
                  )}
                  
                  {currentJob.status === 'completed' && currentJob.documents && (
                    <div className="completion-summary">
                      <div className="success-message">
                        ✅ Processing completed successfully!
                      </div>
                      <div className="results-summary">
                        📊 Total chunks created: {currentJob.documents.reduce((sum, doc) => sum + (doc.chunks_created || 0), 0)}
                      </div>
                    </div>
                  )}

                  {currentJob.status === 'failed' && (
                    <div className="error-message">
                      ❌ Processing failed: {currentJob.error || 'Unknown error'}
                    </div>
                  )}
                  
                  {currentJob.status === 'cancelled' && (
                    <div className="cancelled-message">
                      🚫 Processing was cancelled
                    </div>
                  )}
                </div>

                {/* Error Details */}
                {currentJob.errors && currentJob.errors.length > 0 && (
                  <div className="error-details">
                    <h4>⚠️ Errors ({currentJob.errors.length})</h4>
                    {currentJob.errors.slice(0, 3).map((error, index) => (
                      <div key={index} className="error-item">
                        <span className="error-file">{error.file_path.split('/').pop()}</span>
                        <span className="error-message">{error.error}</span>
                      </div>
                    ))}
                    {currentJob.errors.length > 3 && (
                      <div className="error-more">
                        ...and {currentJob.errors.length - 3} more errors
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Section 2: S3 Upload */}
          <div className="section">
            <h2>☁️ S3 Upload & Processing</h2>
            
            <div className="form-group">
              <label>Select File:</label>
              <input
                type="file"
                onChange={(e) => setSelectedFile(e.target.files[0])}
                disabled={s3Processing}
                accept={supportedExtensions.map(ext => ext).join(',')}
              />
              {selectedFile && (
                <div className="file-info">
                  Selected: {selectedFile.name} ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                </div>
              )}
            </div>

            <div className="form-group">
              <label>S3 Path:</label>
              <input
                type="text"
                value={s3Path}
                onChange={(e) => setS3Path(e.target.value)}
                placeholder="documents/folder"
                disabled={s3Processing}
              />
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Chunking Method:</label>
                <select
                  value={s3ChunkingMethod}
                  onChange={(e) => setS3ChunkingMethod(e.target.value)}
                  disabled={s3Processing}
                >
                  {chunkingMethods.map(method => (
                    <option key={method.name} value={method.name}>
                      {method.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="form-group">
                <label>Chunk Size:</label>
                <input
                  type="number"
                  value={s3ChunkSize}
                  onChange={(e) => setS3ChunkSize(parseInt(e.target.value))}
                  min="100"
                  max="4000"
                  disabled={s3Processing}
                />
              </div>

              <div className="form-group">
                <label>Chunk Overlap:</label>
                <input
                  type="number"
                  value={s3ChunkOverlap}
                  onChange={(e) => setS3ChunkOverlap(parseInt(e.target.value))}
                  min="0"
                  max="1000"
                  disabled={s3Processing}
                />
              </div>
            </div>

            <button
              onClick={uploadToS3}
              disabled={s3Processing}
              className="btn btn-primary"
            >
              {s3Processing ? 'Uploading & Processing...' : 'Upload to S3 & Process'}
            </button>

            {/* S3 Processing Progress Bar */}
            {s3Processing && (
              <div className="s3-processing-status">
                <h3>📤 Upload & Processing Status</h3>
                
                <div className="progress-container">
                  <div className="progress-header">
                    <span className="progress-text">
                      {s3ProgressStage || 'Processing...'}
                    </span>
                    <span className="progress-percentage">
                      {s3ProgressPercent}%
                    </span>
                  </div>
                  
                  <div className="progress-bar">
                    <div 
                      className={`progress-fill ${
                        s3ProgressPercent === 100 ? 'completed' : 'processing'
                      }`}
                      style={{ 
                        width: `${s3ProgressPercent}%`,
                        transition: 'width 0.3s ease'
                      }}
                    ></div>
                  </div>
                </div>

                <div className="processing-details">
                  <div className="detail-row">
                    <span className="detail-label">Status:</span>
                    <span className="detail-value">{s3ProgressStage || 'Processing...'}</span>
                  </div>
                  
                  {selectedFile && (
                    <div className="detail-row">
                      <span className="detail-label">File:</span>
                      <span className="detail-value current-file">
                        {selectedFile.name}
                      </span>
                    </div>
                  )}
                  
                  <div className="detail-row">
                    <span className="detail-label">Target S3 Path:</span>
                    <span className="detail-value">
                      {s3Path}
                    </span>
                  </div>
                  
                  {selectedFile && (
                    <div className="detail-row">
                      <span className="detail-label">File Size:</span>
                      <span className="detail-value">
                        {(selectedFile.size / 1024).toFixed(1)} KB
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* S3 Result */}
            {s3Result && (
              <div className={`result-message ${s3Result.status}`}>
                {s3Result.status === 'success' ? (
                  <div>
                    ✅ <strong>Success!</strong> File uploaded and processed.
                    <br />Document ID: {s3Result.document_id}
                    <br />Chunks created: {s3Result.chunks_created}
                    <br />S3 Key: {s3Result.s3_key}
                  </div>
                ) : (
                  <div>
                    ❌ <strong>Error:</strong> {s3Result.message}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Search Section */}
        <div className="section">
          <h2>🔍 Search Documents</h2>
          <div className="search-form">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Enter search query..."
              onKeyPress={(e) => e.key === 'Enter' && searchDocuments()}
            />
            <button onClick={searchDocuments} className="btn btn-primary">
              Search
            </button>
          </div>

          {searchResults && (
            <div className="search-results">
              <h3>Search Results</h3>
              
              {/* Consolidated Answer */}
              {searchResults.response && (
                <div className="search-answer">
                  <h4>📝 Answer</h4>
                  <div className="answer-content">
                    {searchResults.response}
                  </div>
                </div>
              )}
              
              {/* Source Documents */}
              {searchResults.sources && searchResults.sources.length > 0 && (
                <div className="source-documents">
                  <h4>📚 Source Documents ({searchResults.sources.length})</h4>
                  {searchResults.sources.map((source, index) => (
                    <div key={index} className="search-result">
                      <div className="result-header">
                        <strong>{source.s3_key || source.file_name}</strong>
                        <span className="similarity">
                          Score: {((source.combined_score || source.similarity || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="result-meta">
                        Type: {source.search_type} | Chunk: {source.chunk_index}
                      </div>
                      <div className="result-content">
                        {(source.content_preview || source.content || '').substring(0, 200)}...
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Recent Documents */}
        <div className="section">
          <h2>📄 Recent Documents</h2>
          <div className="documents-table">
            {recentDocuments.length > 0 ? (
              <table>
                <thead>
                  <tr>
                    <th>File Name</th>
                    <th>Type</th>
                    <th>Chunking Method</th>
                    <th>Chunks</th>
                    <th>Status</th>
                    <th>Created</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {recentDocuments.map((doc) => (
                    <tr key={doc.id}>
                      <td>{doc.file_name}</td>
                      <td>{doc.source_type}</td>
                      <td>{doc.chunking_method}</td>
                      <td>{doc.actual_chunks || doc.total_chunks}</td>
                      <td className={`status ${doc.processing_status}`}>
                        {doc.processing_status}
                      </td>
                      <td>{new Date(doc.created_at).toLocaleDateString()}</td>
                      <td>
                        <button
                          onClick={() => deleteDocument(doc.id)}
                          className="btn btn-danger btn-sm"
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p>No documents found</p>
            )}
          </div>
        </div>

        {/* Supported Extensions Info */}
        <div className="section">
          <h2>📋 Supported File Types</h2>
          <div className="extensions-list">
            {supportedExtensions.map((ext, index) => (
              <span key={index} className="extension-tag">{ext}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChunkingInterface;
