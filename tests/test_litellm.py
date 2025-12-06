"""
Test LiteLLM integration with Agno

Tests the LiteLLM proxy connection with a simple agent.
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_litellm_config():
    """Test LiteLLM configuration"""
    print("=" * 50)
    print("TEST 1: LiteLLM Configuration")
    print("=" * 50)
    
    base_url = os.getenv("LITELLM_API_BASE")
    api_key = os.getenv("LITELLM_API_KEY")
    
    if not base_url:
        print("❌ LITELLM_API_BASE not set")
        return False
    if not api_key:
        print("❌ LITELLM_API_KEY not set")
        return False
    
    print(f"✅ Base URL: {base_url}")
    print(f"✅ API Key: {api_key[:15]}...")
    return True


def test_litellm_direct():
    """Test LiteLLM direct call"""
    print("\n" + "=" * 50)
    print("TEST 2: LiteLLM Direct Call")
    print("=" * 50)
    
    try:
        from openai import OpenAI
        
        base_url = os.getenv("LITELLM_API_BASE")
        api_key = os.getenv("LITELLM_API_KEY")
        
        # Available models on this proxy:
        # - claude-sonnet-4-5-20250929
        # - claude-haiku-4-5-20251001
        # - claude-opus-4-1-20250805
        # - claude-opus-4-5-20251101
        # - gpt-5-2025-08-07
        # - gpt-5-mini-2025-08-07
        # - gpt-5-nano-2025-08-07
        # - gpt-5-codex
        
        model = "gpt-5-mini-2025-08-07"  # Fast and cheap
        
        print(f"Using model: {model}")
        print("Calling proxy via OpenAI client...")
        
        client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key=api_key,
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'Hello from the proxy!' only."}],
            max_tokens=20,
        )
        content = response.choices[0].message.content
        print(f"✅ Response: {content}")
        return True
        
    except Exception as e:
        print(f"❌ LiteLLM call failed: {e}")
        return False


def test_agno_with_litellm():
    """Test Agno Agent with LiteLLM"""
    print("\n" + "=" * 50)
    print("TEST 3: Agno Agent with LiteLLM")
    print("=" * 50)
    
    try:
        import litellm
        litellm.drop_params = True  # Drop unsupported params
        
        from agno.agent import Agent
        from agno.models.litellm import LiteLLM
        
        base_url = os.getenv("LITELLM_API_BASE")
        api_key = os.getenv("LITELLM_API_KEY")
        
        model = "gpt-5-mini-2025-08-07"
        print(f"Creating Agno Agent with model: {model}")
        
        # Create agent with LiteLLM pointing to proxy
        agent = Agent(
            name="Test Agent",
            model=LiteLLM(
                id=model,
                api_base=f"{base_url}/v1",
                api_key=api_key,
            ),
            instructions=["You are a helpful assistant. Be concise."],
            markdown=True,
        )
        
        print("Running agent...")
        response = agent.run("What is 2 + 2? Answer in one word.")
        
        print(f"✅ Agent Response: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Agno Agent failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agno_with_tools():
    """Test Agno Agent with LiteLLM and our Perplexity tools"""
    print("\n" + "=" * 50)
    print("TEST 4: Agno Agent with Tools")
    print("=" * 50)
    
    try:
        import litellm
        litellm.drop_params = True  # Drop unsupported params
        
        from agno.agent import Agent
        from agno.models.litellm import LiteLLM
        from infrastructure.perplexity_tools import PerplexitySearchTools
        
        base_url = os.getenv("LITELLM_API_BASE")
        api_key = os.getenv("LITELLM_API_KEY")
        
        model = "gpt-5-mini-2025-08-07"
        print(f"Creating Agno Agent with model: {model} + PerplexitySearchTools...")
        
        # Create tools
        search_tools = PerplexitySearchTools(max_results=3)
        
        # Create agent
        agent = Agent(
            name="Research Agent",
            model=LiteLLM(
                id=model,
                api_base=f"{base_url}/v1",
                api_key=api_key,
            ),
            tools=[search_tools],
            instructions=[
                "You are a research assistant.",
                "Use the search tool to find information.",
                "Be concise in your responses.",
            ],
            markdown=True,
        )
        
        print("Running agent with search task...")
        response = agent.run("What is Agno AI framework? Search and give me a 2 sentence summary.")
        
        print(f"\n✅ Agent Response:\n{response.content}")
        return True
        
    except Exception as e:
        print(f"❌ Agno Agent with tools failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all LiteLLM tests"""
    print("\n" + "=" * 60)
    print("  LITELLM + AGNO TEST SUITE")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Config
    results["config"] = test_litellm_config()
    
    if not results["config"]:
        print("\n⚠️  Skipping remaining tests - config not set")
        return False
    
    # Test 2: Direct LiteLLM
    results["litellm_direct"] = test_litellm_direct()
    
    # Test 3: Agno with LiteLLM
    if results["litellm_direct"]:
        results["agno_litellm"] = test_agno_with_litellm()
    
    # Test 4: Agno with Tools
    if results.get("agno_litellm"):
        results["agno_tools"] = test_agno_with_tools()
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"   {status} {name}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n   Total: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()

