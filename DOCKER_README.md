# AWS LLM RAGA Project - Docker Deployment

This guide explains how to deploy the AWS LLM RAGA Project using Docker containers.

## 🐳 Docker Services

The system consists of 4 main services:

1. **aws-llm-raga-frontend** - React UI application
2. **aws-llm-raga-backend** - Flask API with advanced chunking
3. **aws-llm-raga-database** - PostgreSQL with pgvector extension
4. **aws-llm-raga-cache** - Redis for caching and job queues

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- Ports 3000, 5000, 5432, 6379 available

### 1. Environment Setup

Copy the environment template:
```bash
cp .env.docker .env
```

Edit `.env` with your configuration:
```bash
# Required: Add your API keys
OPENAI_API_KEY=your_openai_api_key
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=your-bucket-name

# Optional: Customize other settings
CHUNKING_METHOD=adaptive
POSTGRES_PASSWORD=ragpassword123
```

### 2. Deploy

Use the deployment script:
```bash
./deploy.sh deploy
```

Or manually with docker-compose:
```bash
docker-compose up -d --build
```

### 3. Access the Application

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **API Health**: http://localhost:5000/api/health

## 📋 Management Commands

### Using the Deploy Script

```bash
# Deploy all services
./deploy.sh deploy

# Check service status
./deploy.sh status

# View logs
./deploy.sh logs
./deploy.sh logs aws-llm-raga-backend

# Stop services
./deploy.sh stop

# Restart services
./deploy.sh restart

# Clean up old containers/images
./deploy.sh cleanup

# Show help
./deploy.sh help
```

### Using Docker Compose Directly

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart specific service
docker-compose restart aws-llm-raga-backend

# Scale backend (if needed)
docker-compose up -d --scale aws-llm-raga-backend=2
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | Database password | `ragpassword123` |
| `AWS_ACCESS_KEY_ID` | AWS access key | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | Required |
| `AWS_S3_BUCKET` | S3 bucket name | Required |
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `CHUNKING_METHOD` | Chunking strategy | `adaptive` |
| `CHUNK_SIZE` | Chunk size in characters | `1000` |
| `SEMANTIC_THRESHOLD` | Semantic similarity threshold | `0.7` |

### Chunking Methods

- **adaptive**: Automatically chooses best method
- **recursive**: Hierarchical text splitting
- **semantic**: Embedding-based chunking
- **agentic**: LLM-guided chunking

## 🗄️ Data Persistence

### Volumes

- `aws_llm_raga_database_volume`: PostgreSQL data
- `aws_llm_raga_uploads_volume`: Uploaded files
- `aws_llm_raga_cache_volume`: Redis cache data

### Backup Database

```bash
# Create backup
docker-compose exec aws-llm-raga-database pg_dump -U postgres rag_system > backup.sql

# Restore backup
docker-compose exec -T aws-llm-raga-database psql -U postgres rag_system < backup.sql
```

## 🔍 Monitoring

### Health Checks

All services include health checks:
- Frontend: HTTP GET to port 80
- Backend: HTTP GET to `/api/health`
- Database: `pg_isready` command
- Cache: Redis ping command

### Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f aws-llm-raga-backend

# Last 100 lines
docker-compose logs --tail=100 aws-llm-raga-backend
```

### Resource Usage

```bash
# Container stats
docker stats

# Service resource usage
docker-compose top
```

## 🐛 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find process using port
lsof -i :3000
# Kill process if needed
kill -9 <PID>
```

**Database Connection Issues**
```bash
# Check database health
docker-compose exec aws-llm-raga-database pg_isready -U postgres

# Reset database
docker-compose down
docker volume rm aws_llm_raga_database_volume
docker-compose up -d
```

**Out of Memory**
```bash
# Check Docker memory usage
docker system df
docker system prune -f
```

**Image Build Failures**
```bash
# Clean build cache
docker builder prune -f
docker-compose build --no-cache
```

### Service Dependencies

Services start in this order:
1. Database (PostgreSQL + pgvector)
2. Cache (Redis)
3. Backend (waits for database)
4. Frontend (waits for backend)

### Performance Tuning

**Backend Scaling**
```bash
# Run multiple backend instances
docker-compose up -d --scale aws-llm-raga-backend=3
```

**Database Optimization**
```bash
# Connect to database
docker-compose exec aws-llm-raga-database psql -U postgres rag_system

# Check table sizes
\dt+

# Analyze query performance
EXPLAIN ANALYZE SELECT ...;
```

## 🔒 Security

### Production Considerations

1. **Change default passwords**
2. **Use Docker secrets for sensitive data**
3. **Enable TLS/SSL certificates**
4. **Configure firewall rules**
5. **Use non-root containers**
6. **Regular security updates**

### Network Security

```bash
# Inspect network
docker network inspect aws-llm-raga-network

# Check exposed ports
docker-compose ps
```

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale backend workers
docker-compose up -d --scale aws-llm-raga-backend=3

# Use load balancer (nginx example)
# Add nginx service to docker-compose.yml
```

### Vertical Scaling

Edit `docker-compose.yml` to add resource limits:
```yaml
services:
  aws-llm-raga-backend:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          memory: 2G
```

## 🆘 Support

### Getting Help

1. Check service logs: `./deploy.sh logs`
2. Verify health checks: `./deploy.sh status`
3. Review configuration: `.env` file
4. Test connectivity: `curl http://localhost:5000/api/health`

### Useful Commands

```bash
# Enter backend container
docker-compose exec aws-llm-raga-backend bash

# Database shell
docker-compose exec aws-llm-raga-database psql -U postgres rag_system

# Redis CLI
docker-compose exec aws-llm-raga-cache redis-cli

# Check file uploads
docker-compose exec aws-llm-raga-backend ls -la /app/uploads/
```

---

🎉 **Your AWS LLM RAGA Project is now running with Docker!**

Visit http://localhost:3000 to start using the advanced chunking interface.
