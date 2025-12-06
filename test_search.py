"""
Basic test script for Perplexity Search Tools

Run this to verify the search functionality works before building the full system.

Usage:
    python test_search.py
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_config():
    """Test configuration loading"""
    print("=" * 50)
    print("TEST 1: Configuration")
    print("=" * 50)
    
    from config import config, validate_config
    
    issues = validate_config()
    if issues:
        print("⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    print("✅ Configuration valid!")
    print(f"   Perplexity API Key: {config.perplexity_api_key[:15]}..." if config.perplexity_api_key else "   No API key")
    return True


def test_perplexity_import():
    """Test perplexity package import"""
    print("\n" + "=" * 50)
    print("TEST 2: Perplexity Package Import")
    print("=" * 50)
    
    try:
        # Package is 'perplexityai' on pip but imports as 'perplexity'
        from perplexity import Perplexity
        print("✅ perplexity package imported successfully (pip: perplexityai)")
        return True
    except ImportError as e:
        print(f"❌ Failed to import perplexity: {e}")
        print("   Run: pip install perplexityai")
        return False


def test_toolkit_import():
    """Test our custom toolkit import"""
    print("\n" + "=" * 50)
    print("TEST 3: PerplexitySearchTools Import")
    print("=" * 50)
    
    try:
        from infrastructure.perplexity_tools import PerplexitySearchTools
        print("✅ PerplexitySearchTools imported successfully")
        
        # Check registered tools
        tools = PerplexitySearchTools()
        print(f"   Registered tools: {[t.__name__ for t in [tools.search, tools.batch_search, tools.search_academic, tools.search_general]]}")
        return True
    except Exception as e:
        print(f"❌ Failed to import PerplexitySearchTools: {e}")
        return False


def test_single_search():
    """Test a single search query"""
    print("\n" + "=" * 50)
    print("TEST 4: Single Search Query")
    print("=" * 50)
    
    from infrastructure.perplexity_tools import PerplexitySearchTools
    
    tools = PerplexitySearchTools(max_results=3)
    
    query = "What is Agno AI framework?"
    print(f"Query: {query}")
    print("-" * 40)
    
    try:
        result = tools.search(query, max_results=3)
        print(result)
        
        # Check for actual API errors (at the start of result)
        if result.startswith("Search error:"):
            print("⚠️  Search returned an error")
            return False
        
        print("\n✅ Single search completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False


def test_batch_search():
    """Test batch search with multiple queries"""
    print("\n" + "=" * 50)
    print("TEST 5: Batch Search (Multi-Query)")
    print("=" * 50)
    
    from infrastructure.perplexity_tools import PerplexitySearchTools
    
    tools = PerplexitySearchTools(max_results=5)
    
    queries = [
        "AI agent frameworks 2024",
        "LangChain vs Agno comparison",
    ]
    print(f"Queries: {queries}")
    print("-" * 40)
    
    try:
        result = tools.batch_search(queries, max_results=3)
        print(result)
        
        # Check for actual API errors (at the start of result)
        if result.startswith("Batch search error:"):
            print("⚠️  Batch search returned an error")
            return False
        
        print("\n✅ Batch search completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Batch search failed: {e}")
        return False


def test_academic_search():
    """Test academic domain filtering"""
    print("\n" + "=" * 50)
    print("TEST 6: Academic Search (Domain Filter)")
    print("=" * 50)
    
    from infrastructure.perplexity_tools import PerplexitySearchTools
    
    tools = PerplexitySearchTools(max_results=5)
    
    query = "transformer architecture deep learning"
    print(f"Query: {query}")
    print("Filter: Academic sources only (arxiv, nature, ieee, etc.)")
    print("-" * 40)
    
    try:
        result = tools.search_academic(query, max_results=3)
        print(result)
        
        print("\n✅ Academic search completed!")
        return True
    except Exception as e:
        print(f"❌ Academic search failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  DEEP RESEARCH SWARM - SEARCH TOOLS TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Config
    results["config"] = test_config()
    
    # Test 2: Perplexity import
    results["perplexity_import"] = test_perplexity_import()
    
    # Test 3: Toolkit import
    results["toolkit_import"] = test_toolkit_import()
    
    # Only run API tests if imports succeeded and Perplexity key is available
    perplexity_key_available = bool(os.getenv("PERPLEXITY_API_KEY"))
    
    if results["perplexity_import"] and results["toolkit_import"] and perplexity_key_available:
        # Test 4: Single search
        results["single_search"] = test_single_search()
        
        # Test 5: Batch search
        results["batch_search"] = test_batch_search()
        
        # Test 6: Academic search
        results["academic_search"] = test_academic_search()
    else:
        if not perplexity_key_available:
            print("\n⚠️  Skipping API tests - PERPLEXITY_API_KEY not set")
        else:
            print("\n⚠️  Skipping API tests due to import failures")
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, passed_test in results.items():
        status = "✅" if passed_test else "❌"
        print(f"   {status} {name}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    # Check for required env var
    if not os.getenv("PERPLEXITY_API_KEY"):
        print("⚠️  PERPLEXITY_API_KEY not found in environment")
        print("   Create a .env file with your API key:")
        print("   PERPLEXITY_API_KEY=pplx-...")
        print("\nContinuing with import tests only...\n")
    
    success = run_all_tests()
    sys.exit(0 if success else 1)

