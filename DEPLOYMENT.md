# Deployment Guide

This project supports multiple deployment configurations:

## 🏠 Local Development

For local development with hot reloading and development features:

```bash
# Copy environment variables
cp .env.example .env
# Edit .env with your actual values

# Start services
docker compose up -d

# Services will be available at:
# - Frontend: http://localhost:3000
# - Backend: http://localhost:5001
# - Backend Health: http://localhost:5001/health
```

## ☁️ AWS ECS Deployment

Automated deployment to AWS ECS using GitHub Actions:

### Prerequisites

1. **AWS Account Setup**
   - AWS Account with appropriate permissions
   - ECR repositories created (auto-created by workflow)
   - CloudFormation stack deployment permissions

2. **GitHub Secrets**
   Set these secrets in your GitHub repository:
   ```
   AWS_ACCESS_KEY_ID          # Your AWS access key
   AWS_SECRET_ACCESS_KEY      # Your AWS secret key
   S3_BUCKET_NAME            # Your S3 bucket name
   GEMINI_API_KEY            # Your Gemini API key
   OPENAI_API_KEY            # Your OpenAI API key (optional)
   ```

3. **AWS Secrets Manager**
   Create these secrets in AWS Secrets Manager:
   - `python-s3-webapp/aws-credentials` with AWS keys
   - `python-s3-webapp/llm-keys` with LLM API keys

### Deployment Options

#### Automatic Deployment
- Pushes to `main` branch automatically trigger ECS deployment
- Uses `docker-compose.ecs.yml` configuration

#### Manual Deployment
- Go to GitHub Actions → "Deploy to AWS ECS"
- Click "Run workflow"
- Choose environment (development/staging)

### Architecture

The AWS deployment creates:
- **VPC** with public subnets across multiple AZs
- **Application Load Balancer** for traffic distribution
- **ECS Fargate cluster** running Docker containers
- **CloudWatch Logs** for monitoring
- **Security Groups** with appropriate access rules

## 🔧 Docker Compose Files

| File | Purpose | Usage |
|------|---------|-------|
| `docker-compose.yml` | Local development | `docker compose up` |
| `docker-compose.ecs.yml` | AWS ECS deployment | Used by GitHub Actions |

## 📊 Monitoring

### Local Development
- Backend logs: `docker compose logs backend`
- Frontend logs: `docker compose logs frontend`
- Health check: `curl http://localhost:5001/health`

### AWS ECS
- CloudWatch Logs: `/ecs/python-s3-webapp`
- ECS Service monitoring in AWS Console
- Application Load Balancer health checks

## 🛠 Troubleshooting

### Local Development Issues
```bash
# Reset containers
docker compose down -v
docker compose up --build

# Check logs
docker compose logs -f

# Clean Docker system
docker system prune -f
```

### AWS Deployment Issues
1. Check GitHub Actions logs for deployment errors
2. Verify AWS secrets are correctly set
3. Check CloudFormation stack events in AWS Console
4. Review ECS service events and task definitions
5. Verify ECR repositories and image pushes

## 🔄 Environment Variables

### Required for All Deployments
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `S3_BUCKET_NAME`
- `PROVIDER` (gemini/openai)
- `MODEL` (e.g., gemini-1.5-flash)
- `GEMINI_API_KEY` or `OPENAI_API_KEY`

### AWS-Specific
- `AWS_REGION` (default: us-east-1)
- `ECR_REGISTRY` (auto-set in workflows)
- `IMAGE_TAG` (auto-set in workflows)

### Frontend-Specific
- `REACT_APP_API_URL` (API endpoint for frontend)

## 📈 Scaling

### Local Development
- Single instance for development

### AWS ECS
- Modify ECS service desired count
- Auto-scaling groups can be configured
- Application Load Balancer handles traffic distribution
# Fresh deployment after stack cleanup - Mon Jul 28 10:11:22 CDT 2025
