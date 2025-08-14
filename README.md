# Python S3 RAG### Advanced Capabilities
- **Semantic Search**: Find documents based on meaning, not just keywords
- **Context-Aware Responses**: LLM responses grounded in actual document content
- **Recursive Chunking**: Intelligent document splitting that respects structure
- **Adaptive Processing**: Automatic chunking strategy selection per document type
- **Document Management**: Index, search, and manage document embeddings
- **Real-time Processing**: Fast content extraction and vector generation
- **Source Attribution**: Responses include source document referencespplication

A modern web application with React frontend and Python Flask backend that allows users to query and analyze files stored in Amazon S3 using **Retrieval-Augmented Generation (RAG)** and **Model Context Protocol (MCP)**. The application uses PostgreSQL with pgvector as a vector database for intelligent document search and context retrieval.

## 🚀 Features

### Core Features
- **RAG-Powered Queries**: Advanced document retrieval using vector similarity search
- **Model Context Protocol (MCP)**: Structured document indexing and retrieval system
- **Vector Database**: PostgreSQL with pgvector for efficient similarity search
- **Multi-format Support**: Analyze PDFs, text files, CSVs, JSON, and more
- **LLM Integration**: Supports both Gemini and OpenAI models
- **Intelligent Chunking**: Smart document splitting with overlapping context
- **Auto-indexing**: Automatic indexing of new S3 documents

### Advanced Capabilities
- **Semantic Search**: Find documents based on meaning, not just keywords
- **Context-Aware Responses**: LLM responses enhanced with relevant document context
- **Document Management**: Index, search, and manage document embeddings
- **Real-time Processing**: Fast content extraction and vector generation
- **Source Attribution**: Responses include source document references

## 🏗 Architecture

### Technology Stack
- **Frontend**: React.js with Material-UI
- **Backend**: Python Flask with async support
- **Vector Database**: PostgreSQL with pgvector extension
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM Providers**: Google Gemini, OpenAI GPT
- **File Processing**: PyPDF2 for PDF text extraction
- **MCP Server**: Custom implementation for document operations
- **Deployment**: Docker containers with multiple configuration options

### RAG Pipeline
1. **Document Ingestion**: S3 files are processed and chunked using recursive or adaptive strategies
2. **Intelligent Chunking**: Text split using hierarchical separators (paragraphs → sentences → words)
3. **Embedding Generation**: Text chunks converted to vector representations
4. **Vector Storage**: Embeddings stored in PostgreSQL with metadata
5. **Query Processing**: User queries converted to embeddings
6. **Similarity Search**: Vector database finds relevant document chunks
7. **Context Assembly**: Retrieved chunks formatted for LLM
8. **Response Generation**: LLM generates response with document context

## 📁 Project Structure

```
python-s3-RAGA-web-app/
├── frontend/                          # React frontend application
│   ├── src/
│   │   ├── App.js                    # Main React component
│   │   ├── App.css                   # Styling
│   │   └── index.js                  # Entry point
│   ├── public/
│   └── Dockerfile                    # Frontend container configuration
├── backend/                          # Python Flask backend
│   ├── src/
│   │   ├── app.py                   # Flask application with RAG endpoints
│   │   └── utils/
│   │       ├── rag_llm_helper.py    # RAG-enabled LLM integration
│   │       ├── vector_db_helper.py  # PostgreSQL vector operations
│   │       ├── document_processor.py # Document chunking and processing
│   │       ├── mcp_rag_server.py    # MCP server implementation
│   │       ├── llm_helper.py        # Legacy LLM helper (backward compatibility)
│   │       └── s3_helper.py         # S3 file operations
│   ├── init_db.py                   # Database initialization script
│   ├── requirements.txt             # Python dependencies (includes RAG libraries)
│   └── Dockerfile                   # Backend container configuration
├── infrastructure/                  # AWS CloudFormation templates
├── .github/workflows/              # GitHub Actions for CI/CD
├── docker-compose.yml              # Local development configuration
├── docker-compose.ecs.yml          # AWS ECS configuration
├── DEPLOYMENT.md                   # Detailed deployment guide
└── LLM_CONFIGURATION.md            # LLM setup and configuration
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- PostgreSQL database with pgvector extension
- AWS S3 bucket with documents
- LLM API key (Gemini or OpenAI)

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd python-s3-RAGA-web-app
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration:
   # - AWS credentials and S3 bucket
   # - PostgreSQL connection details
   # - LLM API keys
   ```

3. **Initialize the vector database**:
   ```bash
   cd backend
   python init_db.py
   ```

4. **Start the application**:
   ```bash
   docker compose up -d
   ```

5. **Index your documents**:
   ```bash
   # Index all S3 documents
   curl -X POST http://localhost:5001/api/rag/index-all
   
   # Or index specific document
   curl -X POST http://localhost:5001/api/rag/index-document \
     -H "Content-Type: application/json" \
     -d '{"s3_key": "your-document.pdf"}'
   ```

6. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5001
   - Health Check: http://localhost:5001/health

## 📖 Deployment Options

This application supports multiple deployment configurations:

- **🏠 Local Development**: Docker Compose for easy local testing  
- **☁️ AWS ECS**: Containerized deployment with auto-scaling

For detailed deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md).

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| **AWS Configuration** |
| `AWS_ACCESS_KEY_ID` | AWS access key | ✅ | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | ✅ | - |
| `S3_BUCKET_NAME` | S3 bucket containing files | ✅ | - |
| **PostgreSQL Vector Database** |
| `POSTGRES_HOST` | PostgreSQL host | ✅ | rag-vector-db.c27sai6q6wxe.us-east-1.rds.amazonaws.com |
| `POSTGRES_PORT` | PostgreSQL port | ✅ | 5432 |
| `POSTGRES_USER` | PostgreSQL username | ✅ | postgres |
| `POSTGRES_PASSWORD` | PostgreSQL password | ✅ | - |
| `POSTGRES_DB` | PostgreSQL database name | ✅ | postgres |
| **LLM Configuration** |
| `PROVIDER` | LLM provider (gemini/openai) | ✅ | gemini |
| `MODEL` | LLM model name | ✅ | gemini-1.5-flash |
| `GEMINI_API_KEY` | Google Gemini API key | Conditional | - |
| `OPENAI_API_KEY` | OpenAI API key | Conditional | - |
| **RAG Configuration** |
| `EMBEDDING_MODEL` | Sentence transformer model | ❌ | sentence-transformers/all-MiniLM-L6-v2 |
| `CHUNK_SIZE` | Document chunk size | ❌ | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap size | ❌ | 200 |
| `MAX_CONTEXT_LENGTH` | Max context for LLM | ❌ | 4000 |
| `USE_RECURSIVE_CHUNKING` | Enable recursive chunking | ❌ | true |

### LLM Configuration

For detailed LLM setup instructions, see [LLM_CONFIGURATION.md](LLM_CONFIGURATION.md).

## � API Endpoints

### Legacy Endpoints (Backward Compatible)
- `GET /api/files` - List S3 files
- `POST /api/upload` - Upload file to S3
- `POST /api/generate` - Generate LLM response
- `POST /api/query-files` - Query files (now uses RAG)

### New RAG Endpoints
- `POST /api/rag/index-document` - Index a specific document
- `POST /api/rag/index-all` - Index all S3 documents
- `POST /api/rag/search` - Vector similarity search
- `POST /api/rag/query` - RAG-powered query with context
- `GET /api/rag/stats` - Get vector database statistics
- `GET /api/rag/documents` - List indexed documents

### Example RAG Query
```bash
curl -X POST http://localhost:5001/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings about machine learning?",
    "similarity_threshold": 0.7,
    "max_chunks": 5
  }'
```

## 💬 Usage Examples

### Traditional Queries (Enhanced with RAG)
- "List me files that contain the word 'cook'"
- "What files discuss machine learning?"
- "Show me all PDF files about project management"

### Advanced RAG Queries
- "What are the key insights about customer behavior across all documents?"
- "Summarize the main findings from the research papers"
- "Compare the methodologies mentioned in the technical documents"
- "What recommendations are made regarding data privacy?"

## 🧪 RAG Testing

### Test Document Indexing
```bash
# Check database statistics
curl http://localhost:5001/api/rag/stats

# List indexed documents
curl http://localhost:5001/api/rag/documents

# Search for similar documents
curl -X POST http://localhost:5001/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "limit": 3}'
```

### Test Vector Database
```bash
cd backend
python init_db.py test
```

## 📚 Chunking Strategies

The system supports multiple document chunking strategies:

### Recursive Chunking (Recommended)
- **Hierarchical Separators**: Uses separators in order of preference:
  1. Double newlines (paragraphs)
  2. Single newlines
  3. Sentence endings (`. `, `! `, `? `)
  4. Clause separators (`;`, `,`)
  5. Word boundaries (spaces)
  6. Character level (last resort)

### Sentence-Based Chunking
- Splits on sentence boundaries
- Maintains sentence integrity
- Good for narrative content

### Adaptive Chunking
- Automatically chooses strategy based on:
  - Document type (PDF, JSON, text)
  - Content length and structure
  - Paragraph density

### Test Chunking Strategies
```bash
cd backend

# Test with sample texts
python test_chunking.py

# Test with your own files
python test_chunking.py --file /path/to/document.pdf

# Test multiple files
python test_chunking.py --file doc1.pdf --file doc2.txt --file report.json

# Test with custom settings
python test_chunking.py --file document.pdf --chunk-size 500 --overlap 100

# Test specific strategy only
python test_chunking.py --file document.txt --strategy recursive
```

## 🛠 Development

### Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.9+ (for local backend development)

### Local Development Setup

```bash
# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies  
cd ../frontend
npm install

# Run without Docker (requires separate terminals)
cd ../backend && python src/app.py
cd ../frontend && npm start
```

## 🧪 Testing

```bash
# Test local build
docker compose build --no-cache
docker compose up -d

# Run health checks
curl http://localhost:5001/health
curl http://localhost:3000

# Clean up
docker compose down -v
```

## 🔍 Monitoring

### Health Checks

- Backend: `GET /health`
- Frontend: Standard HTTP health check

### Logs

```bash
# View logs
docker compose logs -f backend
docker compose logs -f frontend

# AWS ECS logs available in CloudWatch
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally using `docker compose up`
5. Submit a pull request

## 📄 License

This project is open source. See the repository for license details.

## 🆘 Support

For issues and questions:
1. Check the [DEPLOYMENT.md](DEPLOYMENT.md) guide
2. Review GitHub Actions logs for deployment issues
3. Open an issue in the repository