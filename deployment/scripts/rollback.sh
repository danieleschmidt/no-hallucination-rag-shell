#!/bin/bash
set -e

echo "🔄 Rolling back RAG System deployment"

NAMESPACE=${NAMESPACE:-production}

# Get current revision
CURRENT=$(kubectl rollout history deployment/rag-system -n $NAMESPACE --revision=0 | tail -n 1 | awk '{print $1}')
PREVIOUS=$((CURRENT - 1))

echo "📊 Current revision: $CURRENT, rolling back to: $PREVIOUS"

# Perform rollback
kubectl rollout undo deployment/rag-system -n $NAMESPACE --to-revision=$PREVIOUS

# Wait for rollback to complete
echo "⏳ Waiting for rollback to complete..."
kubectl rollout status deployment/rag-system -n $NAMESPACE --timeout=300s

# Verify rollback
echo "✅ Verifying rollback..."
kubectl get pods -n $NAMESPACE -l app=rag-system

echo "🎉 Rollback completed successfully!"
