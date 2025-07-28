FROM python:3.9-slim

# Add build arguments
ARG AWS_REGION
ENV AWS_REGION=${AWS_REGION}

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/src ./src

ENV PYTHONPATH=/app

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--access-logfile", "-", "--access-logformat", "%(h)s %(l)s %(u)s %(t)s \"%(r)s\" %(s)s %(b)s \"%(f)s\" \"%(a)s\"", "src.app:app"]
