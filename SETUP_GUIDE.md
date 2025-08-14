# RAG Chunking System Deployment Guide

## System Overview

This is a comprehensive RAG (Retrieval-Augmented Generation) system with an advanced chunking interface that supports:
- Directory-based file processing with multiple chunking strategies
- S3 file upload and processing
- PostgreSQL with pgvector for vector storage
- Real-time job tracking and progress monitoring
- Advanced search capabilities

## Prerequisites

### System Requirements
- Python 3.8+
- Node.js 16+
- PostgreSQL 13+ with pgvector extension
- AWS S3 bucket (optional, for S3 functionality)

### Required Software
```bash
# Install PostgreSQL (macOS)
brew install postgresql@15
brew install postgresql@15

# Install Node.js
brew install node

# Install Python dependencies (will be done in setup)
```

## Database Setup

### 1. Install pgvector Extension
```bash
# Start PostgreSQL service
brew services start postgresql@15

# Connect to PostgreSQL
psql postgres

# Create database and enable pgvector
CREATE DATABASE rag_system;
\c rag_system;
CREATE EXTENSION vector;
\q
```

### 2. Initialize Database Schema
```bash
# Navigate to backend directory
cd backend

# Run the database schema creation
psql -d rag_system -f database_schema.sql
```

## Environment Configuration

### 1. Backend Environment (.env)
Create a `.env` file in the backend directory:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_system
DB_USER=your_username
DB_PASSWORD=your_password

# AWS S3 Configuration (optional)
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=your-bucket-name
AWS_REGION=us-east-1

# Application Configuration
FLASK_ENV=development
FLASK_DEBUG=True
UPLOAD_FOLDER=./uploads
MAX_CONTENT_LENGTH=16777216  # 16MB

# Chunking Configuration
DEFAULT_CHUNK_SIZE=1000
DEFAULT_CHUNK_OVERLAP=200
EMBEDDING_MODEL=all-MiniLM-L6-v2

# OpenAI Configuration (for agentic chunking)
OPENAI_API_KEY=your_openai_api_key
```

### 2. Frontend Environment (.env)
Create a `.env` file in the frontend directory:

```env
REACT_APP_API_URL=http://localhost:5000
REACT_APP_UPLOAD_CHUNK_SIZE=1048576  # 1MB chunks
```

## Installation Steps

### 1. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install additional dependencies for the new system
pip install psycopg2-binary sentence-transformers transformers torch
pip install openai boto3 flask-cors python-dotenv
```

### 2. Update requirements.txt
Add these new dependencies to `backend/requirements.txt`:

```txt
# Existing dependencies...
psycopg2-binary==2.9.7
sentence-transformers==2.2.2
transformers==4.35.0
torch==2.1.0
openai==1.3.0
boto3==1.29.0
flask-cors==4.0.0
python-dotenv==1.0.0
numpy==1.24.0
scikit-learn==1.3.0
```

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Install additional dependencies
npm install axios
```

### 4. Update package.json
Add axios to `frontend/package.json` dependencies:

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0"
  }
}
```

## File Integration

### 1. Backend Integration

#### Update app.py
Add these imports and integrate the new API:

```python
# Add to existing imports in src/app.py
from utils.enhanced_vector_db_helper import EnhancedVectorDBHelper
from utils.chunking_service import ChunkingService
from utils.chunking_api import create_chunking_api

# Initialize new components after existing code
enhanced_db = EnhancedVectorDBHelper()
chunking_service = ChunkingService(enhanced_db)

# Register the chunking API blueprint
chunking_api_bp = create_chunking_api(chunking_service, enhanced_db)
app.register_blueprint(chunking_api_bp, url_prefix='/api')
```

#### Update existing vector_db_helper.py
The new `enhanced_vector_db_helper.py` extends the existing functionality. You can:
1. Replace the old file with the new one, or
2. Gradually migrate by importing the enhanced version alongside the existing one

### 2. Frontend Integration

#### Update App.js
Add the ChunkingInterface component:

```javascript
// Add to src/App.js
import ChunkingInterface from './ChunkingInterface';
import './ChunkingInterface.css';

function App() {
  return (
    <div className="App">
      <ChunkingInterface />
    </div>
  );
}

export default App;
```

## Running the System

### 1. Start Backend
```bash
cd backend
source venv/bin/activate
python src/app.py
```

The backend will run on `http://localhost:5000`

### 2. Start Frontend
```bash
cd frontend
npm start
```

The frontend will run on `http://localhost:3000`

## Testing the System

### 1. Verify Database Connection
```bash
# Test database connectivity
cd backend
python -c "
from utils.enhanced_vector_db_helper import EnhancedVectorDBHelper
db = EnhancedVectorDBHelper()
print('Database connection successful!')
"
```

### 2. Test Directory Processing
1. Open the frontend at `http://localhost:3000`
2. In the "Directory Processing" section:
   - Enter a directory path (e.g., `/Users/yourusername/Documents/test_files`)
   - Select a chunking method
   - Click "Process Directory"
3. Monitor the job status for completion

### 3. Test S3 Upload (if configured)
1. In the "S3 Upload" section:
   - Select a file
   - Choose chunking method
   - Click "Upload and Process"
2. Check the job status and verify files are processed

### 4. Test Search Functionality
1. Use the search box to find relevant content
2. Verify that results show similarity scores and metadata
3. Check that document management features work

## Production Deployment

### 1. Database Configuration
- Use a production PostgreSQL instance
- Enable SSL connections
- Set up regular backups
- Configure connection pooling

### 2. Backend Deployment
```bash
# Use production WSGI server
pip install gunicorn

# Run with gunicorn
gunicorn --bind 0.0.0.0:5000 src.app:app
```

### 3. Frontend Deployment
```bash
# Build for production
npm run build

# Serve static files with nginx or similar
```

### 4. Docker Deployment (Optional)
The existing `docker-compose.yml` can be extended to include the new services:

```yaml
# Add to existing docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg15
    environment:
      POSTGRES_DB: rag_system
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - ./backend/database_schema.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
```

## Monitoring and Maintenance

### 1. Database Monitoring
- Monitor vector index performance
- Track storage usage
- Set up query performance monitoring

### 2. Application Monitoring
- Monitor job processing times
- Track API response times
- Set up logging for debugging

### 3. Regular Maintenance
- Clean up old job records
- Optimize vector indexes
- Update embedding models as needed

## Troubleshooting

### Common Issues

1. **pgvector Extension Not Found**
   ```bash
   # Reinstall pgvector
   brew install pgvector
   psql -d rag_system -c "CREATE EXTENSION vector;"
   ```

2. **CORS Issues**
   - Ensure `flask-cors` is installed and configured
   - Check that frontend URL is in CORS origins

3. **File Upload Issues**
   - Verify upload directory permissions
   - Check MAX_CONTENT_LENGTH settings
   - Ensure adequate disk space

4. **Embedding Model Download**
   - First run will download the sentence-transformers model
   - Ensure internet connectivity and adequate disk space

### Logs and Debugging
- Backend logs: Check Flask console output
- Frontend logs: Check browser developer console
- Database logs: Check PostgreSQL logs for query issues

## Security Considerations

1. **Database Security**
   - Use strong passwords
   - Enable SSL connections in production
   - Restrict database access to application servers

2. **API Security**
   - Implement rate limiting
   - Add authentication/authorization
   - Validate all input parameters

3. **File Upload Security**
   - Validate file types and sizes
   - Scan uploads for malware
   - Use secure file storage

4. **Environment Variables**
   - Never commit `.env` files
   - Use secure secret management in production
   - Rotate API keys regularly

## Support and Maintenance

For ongoing support:
1. Monitor system performance and resource usage
2. Keep dependencies updated with security patches
3. Regularly backup the database and embeddings
4. Test disaster recovery procedures
5. Monitor for new versions of embedding models

This completes the setup for your comprehensive RAG chunking system with the requested UI features!
