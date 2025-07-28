# LLM Configuration Guide

This application supports multiple LLM providers and can be easily configured via environment variables.

## Supported Providers

### 1. Google Gemini (Default)
- **Provider**: `gemini`
- **API Key**: `GEMINI_API_KEY`
- **Supported Models**: 
  - `gemini-1.5-flash` (recommended)
  - `gemini-1.0-pro-vision-latest`
  - `gemini-pro-vision`

### 2. OpenAI
- **Provider**: `openai`
- **API Key**: `OPENAI_API_KEY`  
- **Supported Models**:
  - `gpt-3.5-turbo`
  - `gpt-4`
  - `gpt-4-turbo`
  - `gpt-4o`
  - `gpt-4o-mini`

## Configuration

### Environment Variables

Update your `.env` file with the following variables:

```bash
# LLM Configuration
PROVIDER=gemini                    # or 'openai'
MODEL=gemini-1.5-flash            # or OpenAI model name

# API Keys
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key
```

### How to Switch Providers

#### Switch to OpenAI:
1. Update `.env` file:
   ```bash
   PROVIDER=openai
   MODEL=gpt-3.5-turbo
   OPENAI_API_KEY=your_openai_api_key_here
   ```

2. Restart the services:
   ```bash
   docker compose restart backend
   ```

#### Switch to Gemini:
1. Update `.env` file:
   ```bash
   PROVIDER=gemini
   MODEL=gemini-1.5-flash
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

2. Restart the services:
   ```bash
   docker compose restart backend
   ```

## Testing the Configuration

Test your configuration with curl:

```bash
# Test the API endpoint
curl -X POST http://localhost:5001/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello! What model are you?"}'
```

## Model-Specific Configuration

### Gemini Models
- **Temperature**: Not configurable (uses default)
- **Max Tokens**: Not configurable (uses default)
- **Input**: Plain text prompts

### OpenAI Models
- **Temperature**: 0.7 (configurable in code)
- **Max Tokens**: 1000 (configurable in code)
- **Input**: Chat completion format

## Troubleshooting

### Common Issues

1. **404 Model Not Found**
   - Check if the model name is correct for your provider
   - Verify your API key has access to the specified model

2. **Authentication Errors**
   - Ensure your API key is valid and active
   - Check that the correct API key is set for the provider

3. **Environment Variable Not Loading**
   - Restart the backend container after changing `.env`
   - Verify the variable is properly exported in docker-compose.yml

### Check Current Configuration

```bash
# Check environment variables in running container
docker exec python-s3-web-app-backend-1 env | grep -E "(PROVIDER|MODEL|.*_API_KEY)"
```

## Adding New Providers

To add a new LLM provider:

1. Install the provider's SDK in `requirements.txt`
2. Add initialization code in `LLMHelper.__init__()`
3. Add response generation method (e.g., `_generate_newprovider_response()`)
4. Update the `generate_response()` method to handle the new provider
5. Add environment variables to `.env` and `docker-compose.yml`

## Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `PROVIDER` | LLM provider to use | No | `gemini` |
| `MODEL` | Model name for the selected provider | Yes | - |
| `GEMINI_API_KEY` | Google Gemini API key | If using Gemini | - |
| `OPENAI_API_KEY` | OpenAI API key | If using OpenAI | - |
