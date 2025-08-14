# External AWS Database Setup Guide

This guide explains how to connect your AWS LLM RAGA Project to an external PostgreSQL database on AWS (like RDS) instead of using the local Docker database.

## 🎯 Prerequisites

### 1. AWS RDS PostgreSQL Database
- PostgreSQL 13+ with pgvector extension support
- Network access from your local machine
- Database user with CREATE privileges

### 2. Local Tools
- Docker and Docker Compose
- PostgreSQL client (`psql`)
- Network connectivity to AWS

## 🚀 Quick Setup

### Step 1: Configure Environment

```bash
# Copy the external database template
cp .env.external .env

# Edit .env with your AWS database details
nano .env
```

Update these essential variables in `.env`:
```bash
# Your RDS endpoint
POSTGRES_HOST=your-rds-instance.cluster-abc123.us-east-1.rds.amazonaws.com
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=rag_system

# Your AWS credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_S3_BUCKET=your-s3-bucket

# Your OpenAI API key
OPENAI_API_KEY=your_openai_api_key
```

### Step 2: Setup Database Schema

```bash
# Install pgvector and create schema
./setup-external-db.sh setup
```

### Step 3: Deploy Application

```bash
# Deploy with external database
./deploy.sh deploy-external
```

## 📋 Detailed Configuration

### AWS RDS Setup

#### 1. Create RDS PostgreSQL Instance
```bash
# Using AWS CLI (optional)
aws rds create-db-instance \
    --db-instance-identifier rag-postgres \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 15.4 \
    --master-username postgres \
    --master-user-password YourSecurePassword123 \
    --allocated-storage 20 \
    --vpc-security-group-ids sg-your-security-group \
    --db-subnet-group-name your-db-subnet-group \
    --publicly-accessible
```

#### 2. Configure Security Group
Allow inbound connections on port 5432 from your IP:
```
Type: PostgreSQL
Protocol: TCP
Port: 5432
Source: Your.IP.Address/32
```

#### 3. Enable pgvector Extension
Connect to your RDS instance and run:
```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `POSTGRES_HOST` | RDS endpoint | `mydb.cluster-xyz.us-east-1.rds.amazonaws.com` |
| `POSTGRES_PORT` | Database port | `5432` |
| `POSTGRES_USER` | Database username | `postgres` |
| `POSTGRES_PASSWORD` | Database password | `YourSecurePassword123` |
| `POSTGRES_DB` | Database name | `rag_system` |
| `AWS_ACCESS_KEY_ID` | AWS access key | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `AWS_S3_BUCKET` | S3 bucket name | `my-rag-documents` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |

## 🔧 Management Commands

### Database Management
```bash
# Test database connection
./setup-external-db.sh test

# Apply schema only
./setup-external-db.sh schema

# Verify setup
./setup-external-db.sh verify

# Show connection info
./setup-external-db.sh info
```

### Application Management
```bash
# Deploy with external database
./deploy.sh deploy-external

# Check service status
./deploy.sh status

# View logs
./deploy.sh logs
./deploy.sh logs aws-llm-raga-backend

# Stop services
./deploy.sh stop

# Restart services
./deploy.sh restart
```

### Manual Database Connection
```bash
# Connect to your database manually
psql -h your-rds-endpoint.amazonaws.com -p 5432 -U postgres -d rag_system
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Connection Timeout
```bash
# Check security groups
aws ec2 describe-security-groups --group-ids sg-your-id

# Test connectivity
telnet your-rds-endpoint.amazonaws.com 5432
```

#### 2. pgvector Extension Issues
```sql
-- Check if pgvector is available
SELECT * FROM pg_available_extensions WHERE name = 'vector';

-- Check current extensions
SELECT * FROM pg_extension;

-- Manual installation
CREATE EXTENSION vector;
```

#### 3. Authentication Failed
- Verify username/password in `.env`
- Check RDS parameter groups for authentication settings
- Ensure database user has proper privileges

#### 4. Schema Creation Errors
```bash
# Check if user has CREATE privileges
psql -h your-host -U postgres -c "SELECT has_database_privilege('postgres', 'rag_system', 'CREATE');"

# Grant privileges if needed
GRANT ALL PRIVILEGES ON DATABASE rag_system TO postgres;
```

### Network Issues

#### Check Connectivity
```bash
# Test DNS resolution
nslookup your-rds-endpoint.amazonaws.com

# Test port connectivity
nc -zv your-rds-endpoint.amazonaws.com 5432

# Test with psql
psql -h your-rds-endpoint.amazonaws.com -p 5432 -U postgres -d postgres -c "SELECT version();"
```

#### AWS VPC Configuration
- Ensure RDS is in a public subnet (for external access)
- Configure route tables properly
- Check NACLs (Network Access Control Lists)

## 📊 Performance Optimization

### Database Configuration
```sql
-- Optimize for vector operations
ALTER DATABASE rag_system SET shared_preload_libraries = 'vector';
ALTER DATABASE rag_system SET max_connections = 100;
ALTER DATABASE rag_system SET shared_buffers = '256MB';
ALTER DATABASE rag_system SET effective_cache_size = '1GB';
```

### Connection Pooling
Consider using connection pooling for production:
```bash
# Add to your .env
DB_POOL_SIZE=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
```

## 🔒 Security Best Practices

### 1. Database Security
- Use strong passwords
- Enable encryption at rest
- Enable encryption in transit (SSL)
- Regular security updates

### 2. Network Security
- Restrict security group access
- Use VPC endpoints when possible
- Consider AWS PrivateLink

### 3. Access Control
```sql
-- Create application-specific user
CREATE USER rag_app WITH PASSWORD 'SecureAppPassword';
GRANT CONNECT ON DATABASE rag_system TO rag_app;
GRANT USAGE ON SCHEMA public TO rag_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag_app;
```

## 📈 Monitoring

### Database Monitoring
```sql
-- Check connection count
SELECT count(*) FROM pg_stat_activity;

-- Monitor query performance
SELECT query, calls, total_time, mean_time 
FROM pg_stat_statements 
ORDER BY total_time DESC 
LIMIT 10;

-- Check database size
SELECT pg_size_pretty(pg_database_size('rag_system'));
```

### Application Monitoring
- Check backend logs: `./deploy.sh logs aws-llm-raga-backend`
- Monitor API health: `curl http://localhost:5000/api/health`
- Check database connections in application logs

## 🔄 Backup and Recovery

### Automated Backups
RDS provides automated backups, but you can also create manual snapshots:
```bash
# Create manual snapshot
aws rds create-db-snapshot \
    --db-instance-identifier rag-postgres \
    --db-snapshot-identifier rag-postgres-backup-$(date +%Y%m%d)
```

### Manual Backup
```bash
# Backup database
pg_dump -h your-rds-endpoint.amazonaws.com -U postgres rag_system > rag_backup.sql

# Restore database
psql -h your-rds-endpoint.amazonaws.com -U postgres rag_system < rag_backup.sql
```

## 📞 Support

### Health Checks
```bash
# Quick health check
./setup-external-db.sh test && echo "Database OK" || echo "Database Issues"

# Application health check
curl -f http://localhost:5000/api/health && echo "API OK" || echo "API Issues"
```

### Log Analysis
```bash
# Backend application logs
./deploy.sh logs aws-llm-raga-backend | grep ERROR

# Database connection logs
./deploy.sh logs aws-llm-raga-backend | grep -i "database\|postgres\|connection"
```

---

🎉 **Your AWS LLM RAGA Project is now connected to your external AWS database!**

Access your application at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
