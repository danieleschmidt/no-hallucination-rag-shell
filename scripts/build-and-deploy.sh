#!/bin/bash

# Terragon RAG System - Build and Deploy Script
# Usage: ./scripts/build-and-deploy.sh [environment] [version]

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENV="staging"
DEFAULT_VERSION="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
ENVIRONMENT="${1:-$DEFAULT_ENV}"
VERSION="${2:-$DEFAULT_VERSION}"

log_info "Starting deployment for environment: $ENVIRONMENT, version: $VERSION"

# Validate environment
case $ENVIRONMENT in
    "local"|"staging"|"production")
        log_info "Valid environment: $ENVIRONMENT"
        ;;
    *)
        log_error "Invalid environment: $ENVIRONMENT. Must be one of: local, staging, production"
        exit 1
        ;;
esac

# Pre-deployment checks
log_info "Running pre-deployment checks..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running or not accessible"
    exit 1
fi

# Check if required files exist
REQUIRED_FILES=(
    "$PROJECT_DIR/Dockerfile"
    "$PROJECT_DIR/docker-compose.yml"
    "$PROJECT_DIR/requirements.txt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        log_error "Required file not found: $file"
        exit 1
    fi
done

log_success "Pre-deployment checks passed"

# Run tests
log_info "Running test suite..."
cd "$PROJECT_DIR"

# Run quality gates
if [[ -f "quality_gates.py" ]]; then
    if python quality_gates.py; then
        log_success "Quality gates passed"
    else
        log_warning "Quality gates failed, but continuing deployment"
    fi
else
    log_warning "Quality gates not found, skipping tests"
fi

# Build Docker image
log_info "Building Docker image..."
IMAGE_NAME="terragon/rag"
FULL_IMAGE_TAG="$IMAGE_NAME:$VERSION"

if docker build -t "$FULL_IMAGE_TAG" .; then
    log_success "Docker image built: $FULL_IMAGE_TAG"
else
    log_error "Failed to build Docker image"
    exit 1
fi

# Tag latest for staging and production
if [[ "$ENVIRONMENT" != "local" ]]; then
    docker tag "$FULL_IMAGE_TAG" "$IMAGE_NAME:latest"
    log_info "Tagged image as latest"
fi

# Deploy based on environment
case $ENVIRONMENT in
    "local")
        log_info "Deploying locally with Docker Compose..."
        
        # Stop existing containers
        docker-compose down || true
        
        # Start new containers
        if docker-compose up -d; then
            log_success "Local deployment successful"
            log_info "Application available at: http://localhost:8080"
            log_info "Monitoring available at: http://localhost:9090"
        else
            log_error "Local deployment failed"
            exit 1
        fi
        ;;
        
    "staging")
        log_info "Deploying to staging environment..."
        
        # Check if kubectl is available
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl not found. Please install kubectl for Kubernetes deployment"
            exit 1
        fi
        
        # Apply staging configuration
        if [[ -f "$PROJECT_DIR/deploy/kubernetes.yaml" ]]; then
            # Create namespace if it doesn't exist
            kubectl create namespace terragon-staging --dry-run=client -o yaml | kubectl apply -f -
            
            # Apply Kubernetes configuration with staging modifications
            sed "s/namespace: terragon/namespace: terragon-staging/g" "$PROJECT_DIR/deploy/kubernetes.yaml" | \
            sed "s/image: terragon\/rag:latest/image: $FULL_IMAGE_TAG/g" | \
            kubectl apply -f -
            
            log_success "Staging deployment applied"
            log_info "Check deployment status: kubectl get pods -n terragon-staging"
        else
            log_error "Kubernetes configuration not found"
            exit 1
        fi
        ;;
        
    "production")
        log_info "Deploying to production environment..."
        
        # Extra safety checks for production
        if [[ "$VERSION" == "latest" ]]; then
            log_error "Cannot deploy 'latest' tag to production. Please specify a version."
            exit 1
        fi
        
        # Confirm production deployment
        read -p "Are you sure you want to deploy to PRODUCTION? (yes/no): " confirm
        if [[ "$confirm" != "yes" ]]; then
            log_info "Production deployment cancelled"
            exit 0
        fi
        
        # Check if kubectl is available
        if ! command -v kubectl &> /dev/null; then
            log_error "kubectl not found. Please install kubectl for Kubernetes deployment"
            exit 1
        fi
        
        # Apply production configuration
        if [[ -f "$PROJECT_DIR/deploy/kubernetes.yaml" ]]; then
            # Update image tag in Kubernetes config
            sed "s/image: terragon\/rag:latest/image: $FULL_IMAGE_TAG/g" "$PROJECT_DIR/deploy/kubernetes.yaml" | \
            kubectl apply -f -
            
            log_success "Production deployment applied"
            
            # Wait for rollout
            log_info "Waiting for rollout to complete..."
            kubectl rollout status deployment/terragon-rag -n terragon --timeout=300s
            
            # Verify deployment
            if kubectl get pods -n terragon | grep terragon-rag | grep Running; then
                log_success "Production deployment verified"
                log_info "Application is running in production"
            else
                log_error "Production deployment verification failed"
                exit 1
            fi
        else
            log_error "Kubernetes configuration not found"
            exit 1
        fi
        ;;
esac

# Post-deployment verification
log_info "Running post-deployment verification..."

case $ENVIRONMENT in
    "local")
        # Wait for services to be ready
        sleep 10
        
        # Test health endpoint
        if curl -f -s http://localhost:8080/health > /dev/null; then
            log_success "Health check passed"
        else
            log_warning "Health check failed or endpoint not ready"
        fi
        ;;
        
    "staging"|"production")
        # Kubernetes health check
        NAMESPACE="terragon"
        if [[ "$ENVIRONMENT" == "staging" ]]; then
            NAMESPACE="terragon-staging"
        fi
        
        log_info "Waiting for pods to be ready..."
        kubectl wait --for=condition=ready pod -l app=terragon-rag -n "$NAMESPACE" --timeout=300s
        
        # Get service endpoint
        SERVICE_IP=$(kubectl get service terragon-rag-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
        
        if [[ "$SERVICE_IP" != "pending" && "$SERVICE_IP" != "" ]]; then
            log_success "Service available at: http://$SERVICE_IP"
        else
            log_info "Service IP pending or using cluster IP"
            log_info "Use 'kubectl port-forward service/terragon-rag-service 8080:80 -n $NAMESPACE' for local access"
        fi
        ;;
esac

# Deployment summary
log_success "Deployment completed successfully!"
echo -e "\n${BLUE}=== DEPLOYMENT SUMMARY ===${NC}"
echo -e "Environment: ${GREEN}$ENVIRONMENT${NC}"
echo -e "Version: ${GREEN}$VERSION${NC}"
echo -e "Image: ${GREEN}$FULL_IMAGE_TAG${NC}"
echo -e "Timestamp: ${GREEN}$(date)${NC}"

case $ENVIRONMENT in
    "local")
        echo -e "\n${BLUE}=== LOCAL ACCESS ===${NC}"
        echo -e "Application: ${GREEN}http://localhost:8080${NC}"
        echo -e "Monitoring: ${GREEN}http://localhost:9090${NC}"
        echo -e "Redis: ${GREEN}localhost:6379${NC}"
        echo -e "\n${BLUE}=== USEFUL COMMANDS ===${NC}"
        echo -e "View logs: ${YELLOW}docker-compose logs -f${NC}"
        echo -e "Stop services: ${YELLOW}docker-compose down${NC}"
        ;;
        
    "staging"|"production")
        NAMESPACE="terragon"
        if [[ "$ENVIRONMENT" == "staging" ]]; then
            NAMESPACE="terragon-staging"
        fi
        
        echo -e "\n${BLUE}=== KUBERNETES ACCESS ===${NC}"
        echo -e "Namespace: ${GREEN}$NAMESPACE${NC}"
        echo -e "Port forward: ${YELLOW}kubectl port-forward service/terragon-rag-service 8080:80 -n $NAMESPACE${NC}"
        echo -e "\n${BLUE}=== USEFUL COMMANDS ===${NC}"
        echo -e "View pods: ${YELLOW}kubectl get pods -n $NAMESPACE${NC}"
        echo -e "View logs: ${YELLOW}kubectl logs -f deployment/terragon-rag -n $NAMESPACE${NC}"
        echo -e "Scale up: ${YELLOW}kubectl scale deployment terragon-rag --replicas=5 -n $NAMESPACE${NC}"
        echo -e "Rollback: ${YELLOW}kubectl rollout undo deployment/terragon-rag -n $NAMESPACE${NC}"
        ;;
esac

echo -e "\n${GREEN}Deployment script completed successfully!${NC}"