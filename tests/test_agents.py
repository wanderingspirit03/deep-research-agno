"""
Test script for Phase 3: Agents Layer

Tests:
1. Planner Agent - Query decomposition
2. Worker Agent - Search and save findings
3. Editor Agent - Report synthesis
4. Full pipeline - Planner → Workers → Editor
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Test 1: Configuration Check
# =============================================================================

def test_config():
    """Test that all required environment variables are set"""
    print("=" * 60)
    print("TEST 1: Configuration Check")
    print("=" * 60)
    
    required = {
        "LITELLM_API_BASE": os.getenv("LITELLM_API_BASE"),
        "LITELLM_API_KEY": os.getenv("LITELLM_API_KEY"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),  # For embeddings
    }
    
    all_set = True
    for name, value in required.items():
        if value:
            print(f"✅ {name}: {value[:15]}...")
        else:
            print(f"❌ {name}: NOT SET")
            all_set = False
    
    return all_set


# =============================================================================
# Test 2: Planner Agent
# =============================================================================

def test_planner_agent():
    """Test the Planner Agent"""
    print("\n" + "=" * 60)
    print("TEST 2: Planner Agent")
    print("=" * 60)
    
    try:
        from agents.planner import PlannerAgent
        
        # Create planner with gpt-5-mini-2025-08-07
        planner = PlannerAgent(
            model_id="gpt-5-mini-2025-08-07",
            max_subtasks=5
        )
        
        # Test query
        test_query = "What are the environmental impacts of electric vehicles compared to traditional cars?"
        
        print(f"\n📋 Planning for:\n   '{test_query}'\n")
        print("-" * 50)
        
        # Create plan
        plan = planner.plan(test_query)
        
        # Validate plan
        assert plan is not None, "Plan should not be None"
        assert len(plan.subtasks) > 0, "Plan should have subtasks"
        assert plan.original_query == test_query, "Original query should be preserved"
        
        # Print plan
        print(planner.plan_to_markdown(plan))
        
        print(f"\n✅ Planner test passed! Generated {len(plan.subtasks)} subtasks")
        return True, plan
        
    except Exception as e:
        print(f"\n❌ Planner test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# =============================================================================
# Test 3: Worker Agent
# =============================================================================

def test_worker_agent(subtask=None):
    """Test the Worker Agent"""
    print("\n" + "=" * 60)
    print("TEST 3: Worker Agent")
    print("=" * 60)
    
    try:
        from agents.worker import WorkerAgent
        from agents.planner import Subtask
        
        # Use provided subtask or create test subtask
        if subtask is None:
            subtask = Subtask(
                id=1,
                query="electric vehicle battery production environmental impact",
                focus="EV battery manufacturing environmental concerns",
                search_type="general",
                priority=1
            )
        
        print(f"\n🔍 Executing subtask:\n   Focus: {subtask.focus}\n   Query: {subtask.query}\n")
        print("-" * 50)
        
        # Create worker with gpt-5-mini-2025-08-07
        worker = WorkerAgent(model_id="gpt-5-mini-2025-08-07")
        
        # Execute subtask
        result = worker.execute_subtask(subtask, worker_id="TEST-01")
        
        # Validate result
        assert result is not None, "Result should not be None"
        
        print(f"\n📝 Worker Response:\n{result[:1000]}...")
        
        print(f"\n✅ Worker test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Worker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 4: Editor Agent
# =============================================================================

def test_editor_agent():
    """Test the Editor Agent"""
    print("\n" + "=" * 60)
    print("TEST 4: Editor Agent")
    print("=" * 60)
    
    try:
        from agents.editor import EditorAgent
        
        # Create editor with gpt-5-mini-2025-08-07
        editor = EditorAgent(model_id="gpt-5-mini-2025-08-07")
        
        test_query = "Environmental impact of electric vehicles"
        
        print(f"\n📊 Generating quick summary for:\n   '{test_query}'\n")
        print("-" * 50)
        
        # Generate quick summary
        summary = editor.quick_summary(test_query)
        
        # Validate result
        assert summary is not None, "Summary should not be None"
        assert len(summary) > 0, "Summary should not be empty"
        
        print(f"\n📝 Editor Summary:\n{summary}")
        
        print(f"\n✅ Editor test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Editor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 5: Full Pipeline
# =============================================================================

def test_full_pipeline():
    """Test the full Planner → Worker → Editor pipeline"""
    print("\n" + "=" * 60)
    print("TEST 5: Full Pipeline")
    print("=" * 60)
    
    try:
        from agents.planner import PlannerAgent
        from agents.worker import WorkerAgent
        from agents.editor import EditorAgent
        from infrastructure.perplexity_tools import PerplexitySearchTools
        from infrastructure.knowledge_tools import KnowledgeTools
        
        # Shared tools
        search_tools = PerplexitySearchTools(max_results=5)
        knowledge_tools = KnowledgeTools(db_path="./test_agents_kb")
        
        # Create agents with gpt-5-mini-2025-08-07
        planner = PlannerAgent(model_id="gpt-5-mini-2025-08-07", max_subtasks=3)
        worker = WorkerAgent(
            model_id="gpt-5-mini-2025-08-07",
            search_tools=search_tools,
            knowledge_tools=knowledge_tools
        )
        editor = EditorAgent(
            model_id="gpt-5-mini-2025-08-07",
            knowledge_tools=knowledge_tools
        )
        
        # Research query
        query = "What are the latest breakthroughs in quantum computing?"
        
        print(f"\n🎯 Research Query:\n   '{query}'\n")
        
        # Phase 1: Planning
        print("-" * 50)
        print("📋 PHASE 1: Planning")
        print("-" * 50)
        
        plan = planner.plan(query)
        print(f"   Generated {len(plan.subtasks)} subtasks:")
        for task in plan.subtasks[:3]:  # Show first 3
            print(f"   - [{task.search_type}] {task.focus}")
        
        # Phase 2: Workers (execute first 2 subtasks only for testing)
        print("\n" + "-" * 50)
        print("🔍 PHASE 2: Worker Execution")
        print("-" * 50)
        
        results = worker.execute_subtasks(plan.subtasks[:2])
        
        completed = sum(1 for r in results if r["status"] == "completed")
        print(f"   Completed: {completed}/{len(results)} subtasks")
        
        # Phase 3: Editor synthesis
        print("\n" + "-" * 50)
        print("📝 PHASE 3: Report Synthesis")
        print("-" * 50)
        
        report = editor.synthesize(query)
        
        print("\n" + "=" * 60)
        print("📄 FINAL REPORT")
        print("=" * 60)
        print(report)
        
        print(f"\n✅ Full pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all agent tests"""
    print("\n" + "=" * 70)
    print("   PHASE 3: AGENTS LAYER TEST SUITE")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Configuration
    results["config"] = test_config()
    
    if not results["config"]:
        print("\n⚠️  Configuration incomplete - stopping tests")
        print("   Please set all required environment variables")
        return False
    
    # Test 2: Planner
    passed, plan = test_planner_agent()
    results["planner"] = passed
    
    if not passed:
        print("\n⚠️  Planner test failed - stopping tests")
        return False
    
    # Test 3: Worker (use first subtask from plan)
    first_subtask = plan.subtasks[0] if plan and plan.subtasks else None
    results["worker"] = test_worker_agent(first_subtask)
    
    if not results["worker"]:
        print("\n⚠️  Worker test failed - continuing with remaining tests")
    
    # Test 4: Editor
    results["editor"] = test_editor_agent()
    
    if not results["editor"]:
        print("\n⚠️  Editor test failed - continuing with remaining tests")
    
    # Test 5: Full pipeline (only if all previous tests passed)
    if results["planner"] and results["worker"] and results["editor"]:
        results["pipeline"] = test_full_pipeline()
    else:
        print("\n⚠️  Skipping full pipeline test due to previous failures")
        results["pipeline"] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("   TEST SUMMARY")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}  {name}")
    
    passed_count = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\n" + "-" * 70)
    print(f"   Total: {passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n🎉 All Phase 3 tests passed! Agents layer is working.")
    else:
        print(f"\n⚠️  {total - passed_count} tests failed. Check the output above.")
    
    return passed_count == total


def run_quick_test():
    """Run a quick test of just the planner (faster)"""
    print("\n" + "=" * 60)
    print("   QUICK TEST: Planner Agent Only")
    print("=" * 60)
    
    if not test_config():
        return False
    
    passed, plan = test_planner_agent()
    return passed


if __name__ == "__main__":
    # Check for quick test flag
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)

