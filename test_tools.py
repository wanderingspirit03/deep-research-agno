"""
Phase 2 Tools Integration Test

Tests all three toolkits:
1. PerplexitySearchTools - Web search with domain filtering
2. DaytonaSandboxTools - Code execution and URL verification
3. KnowledgeTools - Vector storage with LanceDB
"""
import os
import sys
import shutil
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_perplexity_tools():
    """Test Perplexity Search Tools"""
    print("\n" + "=" * 60)
    print("TEST: PerplexitySearchTools")
    print("=" * 60)
    
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        print("⚠️  SKIP: PERPLEXITY_API_KEY not set")
        return None  # None = skipped
    
    try:
        from infrastructure import PerplexitySearchTools
        
        tools = PerplexitySearchTools()
        print(f"✅ Initialized PerplexitySearchTools")
        print(f"   - Academic domains: {len(tools.academic_domains)}")
        print(f"   - Denylist domains: {len(tools.denylist_domains)}")
        
        # Test single search
        print("\n--- Single Search ---")
        result = tools.search("what is LangChain framework", max_results=2)
        print(result[:500] + "..." if len(result) > 500 else result)
        
        print("\n✅ PerplexitySearchTools: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ PerplexitySearchTools: FAILED - {e}")
        return False


def test_docker_sandbox_tools():
    """Test Docker Sandbox Tools (local development)"""
    print("\n" + "=" * 60)
    print("TEST: DockerSandboxTools")
    print("=" * 60)
    
    # Check Docker availability
    try:
        import docker
        client = docker.from_env()
        client.ping()
        print("✅ Docker is running")
    except ImportError:
        print("⚠️  SKIP: docker package not installed. Run: pip install docker")
        return None
    except Exception as e:
        print(f"⚠️  SKIP: Docker not available - {e}")
        print("   Make sure Docker Desktop is running")
        return None
    
    try:
        from infrastructure import DockerSandboxTools
        
        tools = DockerSandboxTools()
        print(f"✅ Initialized DockerSandboxTools")
        print(f"   - Image: {tools.image}")
        print(f"   - Memory limit: {tools.memory_limit}")
        
        # Test code execution
        print("\n--- Code Execution ---")
        result = tools.run_code('print("Hello from Docker sandbox!")')
        print(result)
        
        # Test URL verification
        print("\n--- URL Verification ---")
        result = tools.verify_url("https://example.com")
        print(result)
        
        print("\n✅ DockerSandboxTools: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ DockerSandboxTools: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_daytona_tools():
    """Test Daytona Sandbox Tools (cloud)"""
    print("\n" + "=" * 60)
    print("TEST: DaytonaSandboxTools")
    print("=" * 60)
    
    api_key = os.getenv("DAYTONA_API_KEY")
    if not api_key:
        print("⚠️  SKIP: DAYTONA_API_KEY not set")
        return None  # None = skipped
    
    try:
        from infrastructure import DaytonaSandboxTools
        
        tools = DaytonaSandboxTools(auto_cleanup=True)
        print(f"✅ Initialized DaytonaSandboxTools")
        
        # Test code execution
        print("\n--- Code Execution ---")
        result = tools.run_code('print("Hello from Daytona sandbox!")')
        print(result)
        
        # Test URL verification
        print("\n--- URL Verification ---")
        result = tools.verify_url("https://example.com")
        print(result)
        
        # Cleanup
        print("\n--- Cleanup ---")
        result = tools.cleanup()
        print(result)
        
        print("\n✅ DaytonaSandboxTools: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ DaytonaSandboxTools: FAILED - {e}")
        return False


def test_knowledge_tools():
    """Test Knowledge Tools with LanceDB"""
    print("\n" + "=" * 60)
    print("TEST: KnowledgeTools")
    print("=" * 60)
    
    # Check for embedding API
    openai_key = os.getenv("OPENAI_API_KEY")
    litellm_key = os.getenv("LITELLM_API_KEY")
    
    if not openai_key and not litellm_key:
        print("⚠️  SKIP: No embedding API key (OPENAI_API_KEY or LITELLM_API_KEY)")
        return None  # None = skipped
    
    # Use temp database path
    test_db_path = "./test_kb_temp"
    
    try:
        from infrastructure import KnowledgeTools
        
        # Cleanup any existing test DB
        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)
        
        tools = KnowledgeTools(db_path=test_db_path)
        print(f"✅ Initialized KnowledgeTools")
        print(f"   - DB path: {tools.db_path}")
        print(f"   - Embedding model: {tools.embedding_model}")
        
        # Test save_finding
        print("\n--- Save Finding ---")
        result = tools.save_finding(
            content="Transformer architecture uses self-attention mechanisms to process sequences in parallel, achieving state-of-the-art results in NLP tasks.",
            source_url="https://arxiv.org/abs/1706.03762",
            source_title="Attention Is All You Need",
            search_type="academic",
            verified=True,
            subtask_id=1,
            worker_id="test_worker",
        )
        print(result)
        
        # Save another finding
        result = tools.save_finding(
            content="BERT introduced bidirectional pre-training for language understanding, achieving new benchmarks on GLUE and SQuAD.",
            source_url="https://arxiv.org/abs/1810.04805",
            source_title="BERT: Pre-training of Deep Bidirectional Transformers",
            search_type="academic",
            verified=True,
            subtask_id=1,
            worker_id="test_worker",
        )
        
        # Test search_knowledge
        print("\n--- Search Knowledge ---")
        result = tools.search_knowledge("attention mechanism transformer")
        print(result)
        
        # Test list_sources
        print("\n--- List Sources ---")
        result = tools.list_sources()
        print(result)
        
        # Cleanup test DB
        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)
        
        print("\n✅ KnowledgeTools: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ KnowledgeTools: FAILED - {e}")
        # Cleanup on failure
        if os.path.exists(test_db_path):
            shutil.rmtree(test_db_path)
        return False


def main():
    """Run all Phase 2 tests"""
    print("\n" + "=" * 60)
    print("PHASE 2 TOOLS INTEGRATION TEST")
    print("=" * 60)
    
    results = {
        "PerplexitySearchTools": test_perplexity_tools(),
        "DockerSandboxTools": test_docker_sandbox_tools(),
        "DaytonaSandboxTools": test_daytona_tools(),
        "KnowledgeTools": test_knowledge_tools(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, result in results.items():
        if result is True:
            print(f"✅ {name}: PASSED")
            passed += 1
        elif result is None:
            print(f"⚠️  {name}: SKIPPED (missing API key)")
            skipped += 1
        else:
            print(f"❌ {name}: FAILED")
            failed += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    # Success if no failures (skipped is OK)
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

