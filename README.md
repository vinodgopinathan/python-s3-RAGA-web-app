# Python S3 Web Application

A modern web application with React frontend and Python Flask backend that allows users to query and analyze files stored in Amazon S3 using Large Language Models (LLMs). The application supports natural language queries and can extract text from various file formats including PDFs.

## 🚀 Features

- **Natural Language Queries**: Ask questions about your S3 files in plain English
- **Multi-format Support**: Analyze PDFs, text files, CSVs, and more
- **LLM Integration**: Supports both Gemini and OpenAI models
- **Real-time Processing**: Fast file content extraction and analysis
- **Full Prompt Transparency**: See exactly what prompt is sent to the LLM
- **Multiple Deployment Options**: Local development and AWS ECS

## 🏗 Architecture

- **Frontend**: React.js with Material-UI
- **Backend**: Python Flask with AWS S3 integration
- **LLM Providers**: Google Gemini, OpenAI GPT
- **File Processing**: PyPDF2 for PDF text extraction
- **Deployment**: Docker containers with multiple configuration options

## 📁 Project Structure

```
python-s3-web-app/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # Styling
│   │   └── index.js         # Entry point
│   ├── public/
│   └── Dockerfile           # Frontend container configuration
├── backend/                 # Python Flask backend
│   ├── src/
│   │   ├── app.py          # Flask application
│   │   └── utils/
│   │       ├── llm_helper.py    # LLM integration and processing
│   │       └── s3_helper.py     # S3 file operations
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile          # Backend container configuration
├── infrastructure/         # AWS CloudFormation templates
├── .github/workflows/      # GitHub Actions for CI/CD
├── docker-compose.yml      # Local development configuration
├── docker-compose.yml     # Development configuration
├── docker-compose.ecs.yml  # AWS ECS configuration
├── DEPLOYMENT.md           # Detailed deployment guide
└── LLM_CONFIGURATION.md    # LLM setup and configuration
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd python-s3-web-app
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your AWS credentials and API keys
   ```

3. **Start the application**:
   ```bash
   docker compose up -d
   ```

4. **Access the application**:
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

| Variable | Description | Required |
|----------|-------------|----------|
| `AWS_ACCESS_KEY_ID` | AWS access key | ✅ |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | ✅ |
| `S3_BUCKET_NAME` | S3 bucket containing files | ✅ |
| `PROVIDER` | LLM provider (gemini/openai) | ✅ |
| `MODEL` | LLM model name | ✅ |
| `GEMINI_API_KEY` | Google Gemini API key | Conditional |
| `OPENAI_API_KEY` | OpenAI API key | Conditional |

### LLM Configuration

For detailed LLM setup instructions, see [LLM_CONFIGURATION.md](LLM_CONFIGURATION.md).

## 💬 Usage Examples

Try these natural language queries:

- "List me files that contain the word 'cook'"
- "What files discuss machine learning?"
- "Show me all PDF files about project management"
- "Find documents related to data analysis"

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