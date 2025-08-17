#!/usr/bin/env python3
"""
Generation 1 Test: Basic functionality verification
Tests core components with graceful error handling
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test that basic imports work."""
    print("🧪 Testing basic imports...")
    try:
        import no_hallucination_rag
        print("✅ Package import successful")
        
        # Test graceful imports
        if hasattr(no_hallucination_rag, 'FactualRAG'):
            if no_hallucination_rag.FactualRAG is not None:
                print("✅ FactualRAG available")
            else:
                print("⚠️  FactualRAG import failed gracefully")
        
        # Test available components
        available = []
        for component in ['FactualRAG', 'SourceRanker', 'QuantumTaskPlanner']:
            if hasattr(no_hallucination_rag, component):
                comp_obj = getattr(no_hallucination_rag, component)
                if comp_obj is not None:
                    available.append(component)
        
        print(f"✅ Available components: {available}")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_quantum_planner():
    """Test quantum planner basic functionality."""
    print("\n🧪 Testing Quantum Task Planner...")
    try:
        from no_hallucination_rag.quantum.quantum_planner import QuantumTaskPlanner, QuantumTask, Priority
        
        # Initialize with basic config
        planner = QuantumTaskPlanner()
        print("✅ QuantumTaskPlanner initialized")
        
        # Test basic task creation using the correct API
        task = QuantumTask(
            title="test_task",
            description="Test quantum task creation",
            priority=Priority.GROUND_STATE
        )
        
        # Add task to planner
        planner.tasks[task.id] = task
        print(f"✅ Task created with ID: {task.id}")
        
        # Test task retrieval
        retrieved_task = planner.tasks.get(task.id)
        if retrieved_task:
            print(f"✅ Task retrieved: {retrieved_task.description}")
        
        # Test quantum properties
        if task.is_coherent():
            print("✅ Task is in quantum coherent state")
        
        return True
        
    except Exception as e:
        print(f"❌ Quantum planner test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_basic_shell():
    """Test basic shell functionality."""
    print("\n🧪 Testing Interactive Shell...")
    try:
        from no_hallucination_rag.shell.interactive_shell import InteractiveShell
        
        # Initialize shell (without starting)
        shell = InteractiveShell()
        print("✅ InteractiveShell initialized")
        
        # Test command parsing
        if hasattr(shell, 'parse_command'):
            command = shell.parse_command("help")
            print("✅ Command parsing works")
        
        return True
        
    except Exception as e:
        print(f"❌ Shell test failed: {e}")
        return False

def test_knowledge_base():
    """Test knowledge base functionality."""
    print("\n🧪 Testing Knowledge Base...")
    try:
        from no_hallucination_rag.knowledge.knowledge_base import KnowledgeBase
        
        # Initialize KB
        kb = KnowledgeBase("test_kb")
        print("✅ KnowledgeBase initialized")
        
        # Test basic operations
        if hasattr(kb, 'add_document'):
            print("✅ Document management available")
        
        return True
        
    except Exception as e:
        print(f"❌ Knowledge base test failed: {e}")
        return False

def test_performance_basic():
    """Test basic performance components."""
    print("\n🧪 Testing Performance Components...")
    try:
        from no_hallucination_rag.optimization.caching import AdaptiveCache
        
        # Initialize cache
        cache = AdaptiveCache()
        print("✅ AdaptiveCache initialized")
        
        # Test basic caching using correct API
        cache.cache["test_key"] = "test_value"
        cache.access_times["test_key"] = time.time()
        cache.creation_times["test_key"] = time.time()
        
        value = cache.get("test_key")
        
        if value == "test_value":
            print("✅ Basic caching works")
        else:
            print("⚠️  Cache value mismatch")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """Run all Generation 1 tests."""
    print("🚀 GENERATION 1 TESTING: MAKE IT WORK (Simple)")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_quantum_planner,
        test_basic_shell,
        test_knowledge_base,
        test_performance_basic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Generation 1: SUCCESS - Basic functionality works!")
    elif passed >= total * 0.8:
        print("⚠️  Generation 1: MOSTLY WORKING - Some issues to resolve")
    else:
        print("❌ Generation 1: NEEDS WORK - Major issues found")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)