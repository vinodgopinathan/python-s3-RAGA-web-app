import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Typography,
  Paper,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import './App.css';

const Root = styled('div')(({ theme }) => ({
  marginTop: theme.spacing(4),
}));

const Title = styled(Typography)(({ theme }) => ({
  marginBottom: theme.spacing(2),
}));

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(2),
}));

const StyledList = styled(List)({
  width: '100%',
});

function App() {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [error, setError] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [llmResponse, setLlmResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [fileQuery, setFileQuery] = useState('');
  const [fileQueryResponse, setFileQueryResponse] = useState('');
  const [lastFileQuery, setLastFileQuery] = useState(''); // Store the query that was sent
  const [isFileQueryLoading, setIsFileQueryLoading] = useState(false);
  const [filePattern, setFilePattern] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [activeTab, setActiveTab] = useState('files');

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

  const fetchFiles = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/files`);
      setFiles(response.data.files);
    } catch (err) {
      setError('Error fetching files from S3');
      console.error('Error:', err);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('Please select a file first');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      await axios.post(`${API_BASE_URL}/api/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setUploadStatus('File uploaded successfully!');
      setSelectedFile(null);
      fetchFiles(); // Refresh the file list
    } catch (error) {
      setUploadStatus(`Error uploading file: ${error.message}`);
    }
  };

  const handleGenerate = async () => {
    if (!prompt) {
      setLlmResponse('Please enter a prompt.');
      return;
    }
    setIsLoading(true);
    setLlmResponse('');
    try {
      const response = await axios.post(`${API_BASE_URL}/api/generate`, { prompt });
      setLlmResponse(response.data.response);
    } catch (error) {
      setLlmResponse(`Error generating response: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileQuery = async () => {
    if (!fileQuery) {
      setFileQueryResponse('Please enter a query.');
      setLastFileQuery(''); // Clear last query if no input
      return;
    }
    
    // Store the original query for reference
    const queryToSend = fileQuery.trim();
    console.log('🔍 Original fileQuery:', fileQuery);
    console.log('🔍 Query to send:', queryToSend);
    
    // Force a render by setting loading state
    setIsFileQueryLoading(true);
    setFileQueryResponse('Processing your query...');
    setLastFileQuery('Processing...'); // Temporary value while loading
    
    // Add a small delay to ensure state is updated
    await new Promise(resolve => setTimeout(resolve, 100));
    
    try {
      const requestData = {
        query: queryToSend,
        file_pattern: filePattern || undefined,
        max_files: 5
      };
      console.log('📤 Sending request to backend:', requestData);
      const response = await axios.post(`${API_BASE_URL}/api/query-files`, requestData);
      console.log('📥 Response received:', response.data);
      
      // Set the actual full prompt that was sent to the LLM
      const fullPrompt = response.data.full_prompt || queryToSend;
      setLastFileQuery(fullPrompt);
      console.log('🔍 Full LLM prompt set to:', fullPrompt);
      
      setFileQueryResponse(response.data.response);
    } catch (error) {
      console.error('❌ Error:', error);
      setFileQueryResponse(`Error querying files: ${error.message}`);
      setLastFileQuery(queryToSend); // Fallback to original query on error
    } finally {
      setIsFileQueryLoading(false);
    }
  };

  const handleSearchFiles = async () => {
    if (!searchQuery) {
      setSearchResults([]);
      return;
    }
    try {
      const response = await axios.post(`${API_BASE_URL}/api/search-files`, { 
        query: searchQuery 
      });
      setSearchResults(response.data.files);
    } catch (error) {
      console.error('Error searching files:', error);
      setSearchResults([]);
    }
  };

  const handleFileContentView = async (fileKey) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/api/file-content/${encodeURIComponent(fileKey)}`);
      alert(`Content of ${fileKey}:\n\n${response.data.content.substring(0, 500)}...`);
    } catch (error) {
      alert(`Error reading file: ${error.message}`);
    }
  };

  return (
    <Container>
      <Root>
        <Title variant="h4">
          S3 File Browser with LLM Integration
        </Title>
        
        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button 
            className={activeTab === 'files' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('files')}
          >
            File Browser
          </button>
          <button 
            className={activeTab === 'query' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('query')}
          >
            Query Files
          </button>
          <button 
            className={activeTab === 'search' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('search')}
          >
            Search Files
          </button>
          <button 
            className={activeTab === 'llm' ? 'tab active' : 'tab'}
            onClick={() => setActiveTab('llm')}
          >
            LLM Chat
          </button>
        </div>

        {/* File Browser Tab */}
        {activeTab === 'files' && (
          <div>
            <StyledPaper>
              {error ? (
                <Typography color="error">{error}</Typography>
              ) : (
                <StyledList>
                  {files.map((file) => (
                    <ListItem key={file.key}>
                      <ListItemText
                        primary={file.key}
                        secondary={`Last modified: ${new Date(file.last_modified).toLocaleString()}`}
                      />
                      <ListItemSecondaryAction>
                        <button 
                          onClick={() => handleFileContentView(file.key)}
                          className="view-button"
                        >
                          View
                        </button>
                        <Typography variant="body2" color="textSecondary">
                          {formatBytes(file.size)}
                        </Typography>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </StyledList>
              )}
            </StyledPaper>
            <div className="upload-section">
              <input
                type="file"
                onChange={handleFileSelect}
                className="file-input"
              />
              <button onClick={handleUpload} className="upload-button">
                Upload to S3
              </button>
              {uploadStatus && <p className="status-message">{uploadStatus}</p>}
            </div>
          </div>
        )}

        {/* Query Files Tab */}
        {activeTab === 'query' && (
          <div className="query-section">
            <Title variant="h5">Query Files with LLM</Title>
            <p>Ask questions about your files and get intelligent responses based on their content.</p>
            
            <div className="query-inputs">
              <textarea
                className="prompt-input"
                value={fileQuery}
                onChange={(e) => setFileQuery(e.target.value)}
                placeholder="Ask a question about your files (e.g., 'What are the main topics in my documents?', 'Find all references to machine learning')"
              />
              <input
                type="text"
                className="file-pattern-input"
                value={filePattern}
                onChange={(e) => setFilePattern(e.target.value)}
                placeholder="File extensions (optional, e.g., .txt,.md,.pdf)"
              />
              <button 
                onClick={handleFileQuery} 
                className="generate-button" 
                disabled={isFileQueryLoading}
              >
                {isFileQueryLoading ? 'Analyzing Files...' : 'Query Files'}
              </button>
            </div>
            
            {/* Enhanced Debug information */}
            <div style={{ 
              margin: '20px 0', 
              padding: '15px', 
              backgroundColor: '#ffcccc', 
              border: '3px solid #ff0000',
              borderRadius: '8px',
              fontSize: '16px',
              fontWeight: 'bold'
            }}>
              <h3 style={{ color: '#ff0000', margin: '0 0 10px 0' }}>🔧 DEBUG INFORMATION</h3>
              <div>📝 Current Input (fileQuery): "{fileQuery}"</div>
              <div>💾 Full LLM Prompt (lastFileQuery): "{lastFileQuery}"</div>
              <div>📏 Input Length: {fileQuery ? fileQuery.length : 0}</div>
              <div>📏 Full Prompt Length: {lastFileQuery ? lastFileQuery.length : 0}</div>
              <div>🔄 Loading State: {isFileQueryLoading ? 'YES' : 'NO'}</div>
              <div>📤 Response Present: {fileQueryResponse ? 'YES' : 'NO'}</div>
            </div>
            
            {/* Simple query display without complex conditions */}
            {lastFileQuery && (
              <div className="response-section" style={{ marginBottom: '20px' }}>
                <div className="query-display">
                  <Typography variant="h6" style={{ color: '#1976d2', marginBottom: '8px' }}>
                    📤 Full Prompt Sent to LLM:
                  </Typography>
                  <Paper className="query-paper" style={{ marginBottom: '16px', backgroundColor: '#e3f2fd', border: '2px solid #1976d2' }}>
                    <Typography style={{ whiteSpace: 'pre-wrap', padding: '12px', fontStyle: 'italic', fontSize: '14px', fontFamily: 'monospace' }}>
                      {lastFileQuery}
                    </Typography>
                  </Paper>
                </div>
              </div>
            )}

            {fileQueryResponse && (
              <div className="response-section">
                <div className="response-display">
                  <Typography variant="h6" style={{ color: '#2e7d32', marginBottom: '8px' }}>
                    📥 LLM Response:
                  </Typography>
                  <Paper className="response-paper" style={{ backgroundColor: '#e8f5e8', border: '2px solid #2e7d32' }}>
                    <Typography style={{ whiteSpace: 'pre-wrap', padding: '12px' }}>
                      {fileQueryResponse}
                    </Typography>
                  </Paper>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Search Files Tab */}
        {activeTab === 'search' && (
          <div className="search-section">
            <Title variant="h5">Search Files</Title>
            <p>Search for files by name or content.</p>
            
            <div className="search-inputs">
              <input
                type="text"
                className="search-input"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search for files by name or content"
              />
              <button onClick={handleSearchFiles} className="search-button">
                Search
              </button>
            </div>
            
            {searchResults.length > 0 && (
              <div className="search-results">
                <Typography variant="h6">Search Results:</Typography>
                <StyledList>
                  {searchResults.map((file) => (
                    <ListItem key={file.key}>
                      <ListItemText
                        primary={file.key}
                        secondary={`${file.match_type === 'content' ? 'Content match' : 'Filename match'} | ${new Date(file.last_modified).toLocaleString()}`}
                      />
                      <ListItemSecondaryAction>
                        <button 
                          onClick={() => handleFileContentView(file.key)}
                          className="view-button"
                        >
                          View
                        </button>
                        <Typography variant="body2" color="textSecondary">
                          {formatBytes(file.size)}
                        </Typography>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </StyledList>
              </div>
            )}
          </div>
        )}

        {/* LLM Chat Tab */}
        {activeTab === 'llm' && (
          <div className="llm-section">
            <Title variant="h5">LLM Chat</Title>
            <p>Chat with the LLM without file context.</p>
            
            <textarea
              className="prompt-input"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt for the LLM"
            />
            <button onClick={handleGenerate} className="generate-button" disabled={isLoading}>
              {isLoading ? 'Generating...' : 'Generate Response'}
            </button>
            {llmResponse && (
              <div className="response-section">
                <Typography variant="h6">Response:</Typography>
                <Paper className="response-paper">
                  <Typography style={{ whiteSpace: 'pre-wrap' }}>{llmResponse}</Typography>
                </Paper>
              </div>
            )}
          </div>
        )}
      </Root>
    </Container>
  );
}

export default App;
