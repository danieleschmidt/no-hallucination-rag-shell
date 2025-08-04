#!/bin/bash
# Deployment script for No-Hallucination RAG Shell

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
CONFIG_FILE="configs/${ENVIRONMENT}.yaml"
DOCKER_COMPOSE_FILE="docker/docker-compose.yml"

echo -e "${BLUE}🛡️ No-Hallucination RAG Shell Deployment${NC}"
echo "Environment: $ENVIRONMENT"
echo "Config: $CONFIG_FILE"
echo ""

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Configuration file $CONFIG_FILE not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites satisfied${NC}"

# Create necessary directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p data/{knowledge_bases,cache,logs}
mkdir -p models/{factuality,governance,embeddings}
mkdir -p logs

# Set permissions
chmod 755 data models logs
chmod 644 $CONFIG_FILE

echo -e "${GREEN}✅ Directories created${NC}"

# Run security checks
echo -e "${YELLOW}🔒 Running security checks...${NC}"
python3 -c "
import sys
import os
sys.path.insert(0, '.')
from no_hallucination_rag.core.validation import InputValidator
from no_hallucination_rag.security.security_manager import SecurityManager

# Test security components
validator = InputValidator()
security = SecurityManager()

# Test malicious input detection
test_inputs = [
    '<script>alert(\"xss\")</script>',
    '\"; DROP TABLE users; --',
    '../../../etc/passwd'
]

for test_input in test_inputs:
    result = validator.validate_query(test_input)
    if result.is_valid:
        print(f'❌ Security check failed: {test_input}')
        sys.exit(1)

print('✅ Security validation passed')
"

# Run tests
echo -e "${YELLOW}🧪 Running tests...${NC}"
if ! python3 run_tests.py; then
    echo -e "${RED}❌ Tests failed. Deployment aborted.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All tests passed${NC}"

# Build Docker image
echo -e "${YELLOW}🐳 Building Docker image...${NC}"
docker build -f docker/Dockerfile -t no-hallucination-rag:latest .

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker image built successfully${NC}"

# Deploy with Docker Compose
echo -e "${YELLOW}🚀 Deploying with Docker Compose...${NC}"

# Stop existing containers
docker-compose -f $DOCKER_COMPOSE_FILE down

# Start services
docker-compose -f $DOCKER_COMPOSE_FILE up -d

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker Compose deployment failed${NC}"
    exit 1
fi

# Wait for services to be ready
echo -e "${YELLOW}⏳ Waiting for services to be ready...${NC}"
sleep 10

# Health check
echo -e "${YELLOW}🏥 Running health checks...${NC}"
for i in {1..30}; do
    if docker-compose -f $DOCKER_COMPOSE_FILE exec -T no-hallucination-rag python -c "
from no_hallucination_rag import FactualRAG
rag = FactualRAG(enable_optimization=False, enable_metrics=False)
health = rag.get_system_health()
print('Health:', health['status'])
assert health['status'] == 'healthy'
rag.shutdown()
"; then
        echo -e "${GREEN}✅ Health check passed${NC}"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Health check timeout${NC}"
        docker-compose -f $DOCKER_COMPOSE_FILE logs no-hallucination-rag
        exit 1
    fi
    
    echo "Waiting for service to be ready... ($i/30)"
    sleep 2
done

# Performance test
echo -e "${YELLOW}⚡ Running performance test...${NC}"
docker-compose -f $DOCKER_COMPOSE_FILE exec -T no-hallucination-rag python -c "
import time
from no_hallucination_rag import FactualRAG

rag = FactualRAG(enable_optimization=False, enable_metrics=False)

# Test query performance
start_time = time.time()
response = rag.query('What are AI safety requirements?')
end_time = time.time()

response_time = end_time - start_time
print(f'Response time: {response_time:.2f}s')
print(f'Factuality score: {response.factuality_score:.2f}')
print(f'Sources: {len(response.sources)}')

if response_time > 30:
    print('❌ Performance test failed: Response too slow')
    exit(1)

rag.shutdown()
print('✅ Performance test passed')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Performance test failed${NC}"
    exit 1
fi

# Display deployment summary
echo -e "${GREEN}"
echo "🎉 Deployment completed successfully!"
echo ""
echo "Services:"
echo "- No-Hallucination RAG Shell: http://localhost:8000"
echo "- Prometheus (optional): http://localhost:9090"
echo "- Grafana (optional): http://localhost:3000 (admin/admin)"
echo ""
echo "Logs:"
echo "  docker-compose -f $DOCKER_COMPOSE_FILE logs -f no-hallucination-rag"
echo ""
echo "Management:"
echo "  docker-compose -f $DOCKER_COMPOSE_FILE stop      # Stop services"
echo "  docker-compose -f $DOCKER_COMPOSE_FILE start     # Start services"
echo "  docker-compose -f $DOCKER_COMPOSE_FILE restart   # Restart services"
echo "  docker-compose -f $DOCKER_COMPOSE_FILE down      # Remove containers"
echo ""
echo -e "${NC}"