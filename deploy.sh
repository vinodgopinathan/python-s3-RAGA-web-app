#!/bin/bash

# AWS LLM RAGA Project - Docker Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose not found. Please install docker-compose."
        exit 1
    fi
    print_success "docker-compose is available"
}

# Function to clean up old containers and images
cleanup_old() {
    print_info "Cleaning up old containers and images..."
    
    # Stop and remove old containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Remove old images
    docker images | grep -E "(python-s3-webapp|aws-llm-raga)" | awk '{print $3}' | xargs -r docker rmi -f 2>/dev/null || true
    
    # Remove unused volumes
    docker volume prune -f
    
    print_success "Cleanup completed"
}

# Function to build and start services
deploy() {
    print_info "Building and deploying AWS LLM RAGA services..."
    
    # Build images
    print_info "Building Docker images..."
    docker-compose build --no-cache
    
    # Start services
    print_info "Starting services..."
    docker-compose up -d
    
    # Wait for services to be healthy
    print_info "Waiting for services to be ready..."
    
    # Wait for database
    print_info "Waiting for database to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose exec -T aws-llm-raga-database pg_isready -U postgres -d rag_system > /dev/null 2>&1; then
            print_success "Database is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Database did not become ready in time"
        exit 1
    fi
    
    # Wait for backend
    print_info "Waiting for backend API to be ready..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:5000/api/health > /dev/null 2>&1; then
            print_success "Backend API is ready"
            break
        fi
        sleep 3
        timeout=$((timeout - 3))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Backend API did not become ready in time"
        exit 1
    fi
    
    # Wait for frontend
    print_info "Waiting for frontend to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:3000 > /dev/null 2>&1; then
            print_success "Frontend is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Frontend did not become ready in time"
        exit 1
    fi
    
    print_success "All services are ready!"
}

# Function to show service status
show_status() {
    print_info "Service Status:"
    docker-compose ps
    
    echo ""
    print_info "Service URLs:"
    echo "  Frontend (React UI): http://localhost:3000"
    echo "  Backend API: http://localhost:5000"
    echo "  Database: localhost:5432"
    echo "  Redis Cache: localhost:6379"
}

# Function to show logs
show_logs() {
    if [ -n "$1" ]; then
        docker-compose logs -f "$1"
    else
        docker-compose logs -f
    fi
}

# Function to stop services
stop_services() {
    print_info "Stopping AWS LLM RAGA services..."
    docker-compose down
    print_success "Services stopped"
}

# Function to restart services
restart_services() {
    print_info "Restarting AWS LLM RAGA services..."
    docker-compose restart
    print_success "Services restarted"
}

# Function to show help
show_help() {
    echo "AWS LLM RAGA Project - Docker Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  deploy         - Build and deploy all services (with local database)"
    echo "  deploy-external - Deploy with external database connection"
    echo "  status         - Show service status and URLs"
    echo "  logs           - Show logs for all services"
    echo "  logs SERVICE   - Show logs for specific service"
    echo "  stop           - Stop all services"
    echo "  restart        - Restart all services"
    echo "  cleanup        - Remove old containers and images"
    echo "  help           - Show this help message"
    echo ""
    echo "Services:"
    echo "  aws-llm-raga-frontend  - React UI"
    echo "  aws-llm-raga-backend   - Flask API"
    echo "  aws-llm-raga-database  - PostgreSQL DB (local only)"
    echo "  aws-llm-raga-cache     - Redis Cache"
    echo ""
    echo "External Database Usage:"
    echo "  1. Create .env from .env.external template"
    echo "  2. Configure your AWS database details"
    echo "  3. Run: ./setup-external-db.sh setup"
    echo "  4. Run: ./deploy.sh deploy-external"
}

# Function to deploy with external database
deploy_external() {
    print_info "Deploying AWS LLM RAGA services with external database..."
    
    # Check if .env file exists
    if [ ! -f ".env" ]; then
        print_error ".env file not found"
        print_info "Please create .env file from .env.external template:"
        print_info "  cp .env.external .env"
        print_info "  # Edit .env with your database configuration"
        exit 1
    fi
    
    # Build images
    print_info "Building Docker images..."
    docker-compose -f docker-compose.external.yml build --no-cache
    
    # Start services
    print_info "Starting services..."
    docker-compose -f docker-compose.external.yml up -d
    
    # Wait for backend to be ready
    print_info "Waiting for backend API to be ready..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:5000/api/health > /dev/null 2>&1; then
            print_success "Backend API is ready"
            break
        fi
        sleep 3
        timeout=$((timeout - 3))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Backend API did not become ready in time"
        print_info "Check logs with: ./deploy.sh logs aws-llm-raga-backend"
        exit 1
    fi
    
    # Wait for frontend
    print_info "Waiting for frontend to be ready..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:3000 > /dev/null 2>&1; then
            print_success "Frontend is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "Frontend did not become ready in time"
        exit 1
    fi
    
    print_success "All services are ready!"
}

# Main script logic
case "${1:-deploy}" in
    "deploy")
        check_docker
        check_docker_compose
        cleanup_old
        deploy
        show_status
        ;;
    "deploy-external")
        check_docker
        check_docker_compose
        cleanup_old
        deploy_external
        show_status
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2"
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        restart_services
        ;;
    "cleanup")
        check_docker
        check_docker_compose
        cleanup_old
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
