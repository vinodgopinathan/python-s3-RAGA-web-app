# 🔐 Security Configuration Guide

## ⚠️ CRITICAL SECURITY REQUIREMENTS

### **🚨 Never Commit Credentials to Git**

This project is configured to **NEVER** include sensitive credentials in the codebase:

- ✅ All credentials are loaded from environment variables only
- ✅ `.env` files are properly excluded in `.gitignore`
- ✅ No hardcoded API keys or access tokens in source code
- ✅ Test files use environment variables, not placeholders

## 🔧 **Environment Variable Configuration**

### **Required Environment Variables:**

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your_actual_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_actual_aws_secret_access_key
AWS_REGION=us-east-1
AWS_S3_BUCKET=your-s3-bucket-name

# Database Configuration
POSTGRES_HOST=your-postgres-host
POSTGRES_PORT=5432
POSTGRES_USER=your-postgres-user
POSTGRES_PASSWORD=your-postgres-password
POSTGRES_DB=your-database-name

# LLM API Keys
GEMINI_API_KEY=your_actual_gemini_api_key
OPENAI_API_KEY=your_actual_openai_api_key
```

### **How to Set Environment Variables:**

#### **Option 1: Local .env File (Recommended for Development)**
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your actual credentials
nano .env
```

#### **Option 2: Export in Shell (Temporary)**
```bash
export GEMINI_API_KEY=your_actual_api_key
export AWS_ACCESS_KEY_ID=your_actual_access_key
# ... etc
```

#### **Option 3: System Environment (Production)**
Set environment variables through your hosting platform (AWS ECS, Docker, etc.)

## 🛡️ **Security Best Practices**

### **✅ What's Secure:**
- Using `os.environ.get('API_KEY')` to load credentials
- Storing credentials in `.env` files (excluded from git)
- Using IAM roles and policies for AWS access
- Rotating API keys regularly

### **❌ What's Insecure:**
- Hardcoding API keys in source code: `api_key = "sk-abc123"`
- Committing `.env` files to git
- Sharing credentials in chat/email
- Using the same credentials across environments

## 🔍 **Security Verification**

### **Check for Hardcoded Credentials:**
```bash
# Search for potential hardcoded secrets
grep -r "sk-" --include="*.py" .
grep -r "AKIA" --include="*.py" .
grep -r "your_.*_key" --include="*.py" .
```

### **Verify .env Exclusion:**
```bash
# Make sure .env files are not tracked
git status --ignored | grep -E "\.env"
```

## 🚀 **Running Tests Securely**

### **Before Running Test Files:**
```bash
# Set required environment variables
export GEMINI_API_KEY=your_actual_api_key
export AWS_ACCESS_KEY_ID=your_actual_access_key
export AWS_SECRET_ACCESS_KEY=your_actual_secret_key

# Then run tests
python test_chunking_selection.py
```

### **Test Files Will:**
- ✅ Check if environment variables are set
- ✅ Provide helpful error messages if missing
- ✅ Never expose actual credentials in logs

## 🔒 **Production Deployment Security**

### **AWS ECS/Docker:**
```yaml
# docker-compose.yml
environment:
  - GEMINI_API_KEY=${GEMINI_API_KEY}
  - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
  # Load from host environment
```

### **Kubernetes:**
```yaml
# Use secrets
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
data:
  gemini-api-key: <base64-encoded-key>
```

## 🚨 **Emergency Response**

### **If Credentials Are Accidentally Committed:**

1. **Immediately Rotate Keys:**
   - AWS: Deactivate and create new access keys
   - Gemini: Regenerate API key
   - OpenAI: Regenerate API key

2. **Remove from Git History:**
   ```bash
   # Use git filter-branch or BFG Repo-Cleaner
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch .env' \
     --prune-empty --tag-name-filter cat -- --all
   ```

3. **Force Push (Dangerous):**
   ```bash
   git push --force-with-lease origin main
   ```

## ✅ **Security Checklist**

- [ ] All credentials loaded from environment variables
- [ ] `.env` files excluded in `.gitignore`  
- [ ] No hardcoded keys in source code
- [ ] Test files use environment variables
- [ ] Production uses secure credential management
- [ ] API keys rotated regularly
- [ ] Access follows principle of least privilege

## 📞 **Security Contact**

If you discover security vulnerabilities:
1. **Do NOT** open a public issue
2. Contact the maintainer privately
3. Provide details about the vulnerability
4. Allow time for patching before disclosure

---

**Remember: Security is everyone's responsibility! 🛡️**
