#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test imports to verify everything works"""

import sys
from pathlib import Path

# Test 1: Import from benchmark context
print("Test 1: Importing from benchmark.strategies...")
try:
    from benchmark.strategies.rag_gpt.pipeline import RAGGPTPipeline
    print("✅ SUCCESS: benchmark.strategies.rag_gpt.pipeline.RAGGPTPipeline")
except ImportError as e:
    print(f"❌ FAILED: {e}")

# Test 2: Import wrapper
print("\nTest 2: Importing wrapper...")
try:
    from benchmark.strategies.rag_gpt.pipeline import RAGGPTPipeline as Pipeline
    print("✅ SUCCESS: Wrapper can import RAGGPTPipeline")
except ImportError as e:
    print(f"❌ FAILED: {e}")

# Test 3: Import strategy wrapper
print("\nTest 3: Importing RAGWithGPT4oStrategy...")
try:
    from benchmark.strategies._04_rag_gpt import RAGWithGPT4oStrategy
    print("✅ SUCCESS: RAGWithGPT4oStrategy imported")
except ImportError as e:
    print(f"❌ FAILED: {e}")
    try:
        # Try with dot notation
        import benchmark.strategies
        print(f"Available in strategies: {dir(benchmark.strategies)}")
    except:
        pass

print("\n" + "="*60)
print("Import tests complete")
