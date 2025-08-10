#!/bin/bash
set -e

echo "🚀 Deploying RAG System to Production"

# Configuration
NAMESPACE=${NAMESPACE:-production}
IMAGE_TAG=${IMAGE_TAG:-latest}
REPLICAS=${REPLICAS:-3}

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply configurations
echo "📁 Applying ConfigMaps and Secrets..."
kubectl apply -f deployment/kubernetes/ -n $NAMESPACE

# Update image tag in deployment
echo "🐳 Updating image to tag: $IMAGE_TAG"
kubectl set image deployment/rag-system rag-system=terragon/no-hallucination-rag:$IMAGE_TAG -n $NAMESPACE

# Scale deployment
echo "⚖️ Scaling to $REPLICAS replicas..."
kubectl scale deployment rag-system --replicas=$REPLICAS -n $NAMESPACE

# Wait for rollout to complete
echo "⏳ Waiting for deployment to complete..."
kubectl rollout status deployment/rag-system -n $NAMESPACE --timeout=300s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE -l app=rag-system

# Run health check
echo "🏥 Running health check..."
kubectl run health-check --rm -i --restart=Never -n $NAMESPACE --image=curlimages/curl -- \
    curl -f http://rag-system-service/health

echo "🎉 Deployment completed successfully!"
