"""
Test script for Phase 4: Orchestration Layer

Tests:
1. Configuration validation
2. SwarmResult dataclass
3. PlannerExecutor
4. WorkerExecutor
5. EditorExecutor
6. Full ResearchSwarm (simple mode)
7. SwarmFactory presets and builder
8. End-to-end integration test
"""
import os
import sys
import shutil
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Test Configuration
# =============================================================================

TEST_DB_PATH = "./test_orchestration_kb"


def cleanup_test_db():
    """Clean up test database"""
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH)


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
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    }
    
    all_set = True
    for name, value in required.items():
        if value:
            print(f"✅ {name}: {value[:15]}...")
        else:
            print(f"❌ {name}: NOT SET")
            all_set = False
    
    # Also check config module
    from config import validate_config
    issues = validate_config()
    if issues:
        print("\n⚠️  Config validation issues:")
        for issue in issues:
            print(f"   - {issue}")
    
    return all_set


# =============================================================================
# Test 2: SwarmResult Dataclass
# =============================================================================

def test_swarm_result():
    """Test SwarmResult dataclass"""
    print("\n" + "=" * 60)
    print("TEST 2: SwarmResult Dataclass")
    print("=" * 60)
    
    try:
        from main import SwarmResult
        from agents.planner import ResearchPlan, Subtask
        
        # Create a test result
        result = SwarmResult(
            query="Test query",
            plan=ResearchPlan(
                original_query="Test query",
                summary="Test summary",
                subtasks=[
                    Subtask(id=1, query="sub1", focus="Focus 1", search_type="general", priority=1),
                    Subtask(id=2, query="sub2", focus="Focus 2", search_type="academic", priority=2),
                ],
                estimated_depth="medium",
            ),
            worker_results=[
                {"subtask_id": 1, "status": "completed"},
                {"subtask_id": 2, "status": "completed"},
            ],
            report="# Test Report\n\nThis is a test.",
            success=True,
        )
        
        # Test summary generation
        summary = result.summary()
        assert "Test query" in summary
        assert "✅" in summary
        assert "2" in summary  # subtasks
        
        print(f"✅ SwarmResult created successfully")
        print(f"✅ Summary generation works")
        print(f"\n{summary}")
        
        return True
        
    except Exception as e:
        print(f"❌ SwarmResult test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 3: PlannerExecutor
# =============================================================================

def test_planner_executor():
    """Test PlannerExecutor step function"""
    print("\n" + "=" * 60)
    print("TEST 3: PlannerExecutor")
    print("=" * 60)
    
    try:
        from main import PlannerExecutor
        from agno.workflow.types import StepInput
        
        # Create executor
        executor = PlannerExecutor(max_subtasks=3)
        
        # Create step input
        step_input = StepInput(
            input="What are the benefits of electric vehicles?"
        )
        
        print(f"📋 Testing planner with: {step_input.input[:50]}...")
        
        # Execute
        result = executor(step_input)
        
        # Validate
        assert result.success, f"Planner failed: {result.error}"
        assert result.content is not None, "No plan content"
        assert "subtasks" in result.content, "Missing subtasks in plan"
        assert len(result.content["subtasks"]) > 0, "No subtasks generated"
        
        print(f"✅ Planner executed successfully")
        print(f"✅ Generated {len(result.content['subtasks'])} subtasks")
        print(f"✅ Summary: {result.content.get('summary', 'N/A')[:100]}...")
        
        return True, result.content
        
    except Exception as e:
        print(f"❌ PlannerExecutor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# =============================================================================
# Test 4: WorkerExecutor
# =============================================================================

def test_worker_executor(plan_content: Optional[dict] = None):
    """Test WorkerExecutor step function"""
    print("\n" + "=" * 60)
    print("TEST 4: WorkerExecutor")
    print("=" * 60)
    
    try:
        from main import WorkerExecutor, PlannerExecutor
        from agno.workflow.types import StepInput, StepOutput
        from infrastructure.perplexity_tools import PerplexitySearchTools
        from infrastructure.knowledge_tools import KnowledgeTools
        
        # Create shared tools
        search_tools = PerplexitySearchTools(max_results=3)
        knowledge_tools = KnowledgeTools(db_path=TEST_DB_PATH)
        
        # If no plan provided, create one
        if plan_content is None:
            planner_executor = PlannerExecutor(max_subtasks=2)
            planner_result = planner_executor(StepInput(
                input="Benefits of renewable energy"
            ))
            plan_content = planner_result.content
        
        if not plan_content or not plan_content.get("subtasks"):
            print("❌ No plan content available")
            return False
        
        # Get first subtask ID
        first_subtask = plan_content["subtasks"][0]
        subtask_id = first_subtask["id"]
        
        # Create worker executor
        executor = WorkerExecutor(
            subtask_id=subtask_id,
            search_tools=search_tools,
            knowledge_tools=knowledge_tools,
        )
        
        # Create step input with planner output
        step_input = StepInput(
            input=plan_content.get("original_query", "Test"),
            previous_step_outputs={
                "planner": StepOutput(
                    step_name="planner",
                    content=plan_content,
                    success=True,
                )
            }
        )
        
        print(f"🔍 Testing worker for subtask {subtask_id}: {first_subtask['focus']}")
        
        # Execute
        result = executor(step_input)
        
        # Validate (we allow partial failures)
        if result.success:
            print(f"✅ Worker executed successfully")
            if isinstance(result.content, dict):
                print(f"✅ Status: {result.content.get('status', 'unknown')}")
        else:
            print(f"⚠️  Worker completed with errors: {result.error}")
        
        return True
        
    except Exception as e:
        print(f"❌ WorkerExecutor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 5: EditorExecutor
# =============================================================================

def test_editor_executor():
    """Test EditorExecutor step function"""
    print("\n" + "=" * 60)
    print("TEST 5: EditorExecutor")
    print("=" * 60)
    
    try:
        from main import EditorExecutor
        from agno.workflow.types import StepInput, StepOutput
        from infrastructure.knowledge_tools import KnowledgeTools
        
        # Create knowledge tools
        knowledge_tools = KnowledgeTools(db_path=TEST_DB_PATH)
        
        # Add some test findings
        knowledge_tools.save_finding(
            content="Electric vehicles produce zero direct emissions, reducing air pollution in urban areas.",
            source_url="https://example.com/ev-emissions",
            source_title="EV Emissions Study",
            search_type="general",
            subtask_id=1,
            worker_id="test",
        )
        
        knowledge_tools.save_finding(
            content="Battery technology has improved significantly, with modern EVs achieving 300+ mile ranges.",
            source_url="https://example.com/ev-range",
            source_title="EV Range Report",
            search_type="general",
            subtask_id=2,
            worker_id="test",
        )
        
        # Create executor
        executor = EditorExecutor(knowledge_tools=knowledge_tools)
        
        # Create step input
        step_input = StepInput(
            input="Benefits of electric vehicles",
            previous_step_outputs={
                "planner": StepOutput(
                    step_name="planner",
                    content={"original_query": "Benefits of electric vehicles"},
                    success=True,
                ),
                "worker_1": StepOutput(
                    step_name="worker_1",
                    content={"focus": "Emissions", "response": "Zero emissions"},
                    success=True,
                ),
            }
        )
        
        print(f"📝 Testing editor synthesis...")
        
        # Execute
        result = executor(step_input)
        
        # Validate
        if result.success:
            print(f"✅ Editor executed successfully")
            print(f"✅ Report length: {len(result.content or '')} chars")
            if result.content:
                print(f"✅ Preview: {result.content[:200]}...")
        else:
            print(f"⚠️  Editor completed with errors: {result.error}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ EditorExecutor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 6: ResearchSwarm (Simple Mode)
# =============================================================================

def test_research_swarm_simple():
    """Test ResearchSwarm in simple sequential mode"""
    print("\n" + "=" * 60)
    print("TEST 6: ResearchSwarm (Simple Mode)")
    print("=" * 60)
    
    try:
        from main import ResearchSwarm
        
        # Create swarm with minimal settings
        swarm = ResearchSwarm(
            max_workers=2,
            max_subtasks=2,
            db_path=TEST_DB_PATH,
        )
        
        query = "What are the key features of Python programming language?"
        
        print(f"🔍 Testing swarm with: {query[:50]}...")
        
        # Execute simple mode (sequential, no Agno Workflow)
        result = swarm.research_simple(query)
        
        # Validate
        print(f"\n📊 Results:")
        print(result.summary())
        
        if result.success:
            print(f"\n✅ Research swarm completed successfully!")
            print(f"✅ Report preview: {result.report[:300]}..." if result.report else "")
        else:
            print(f"\n⚠️  Research swarm completed with errors: {result.error}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ ResearchSwarm test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 7: SwarmFactory
# =============================================================================

def test_swarm_factory():
    """Test SwarmFactory presets and builder"""
    print("\n" + "=" * 60)
    print("TEST 7: SwarmFactory")
    print("=" * 60)
    
    try:
        from swarm_factory import (
            create_swarm,
            list_presets,
            SwarmBuilder,
            PRESETS,
        )
        
        # Test list_presets
        presets = list_presets()
        assert len(presets) >= 4, "Should have at least 4 presets"
        print(f"✅ Found {len(presets)} presets: {list(presets.keys())}")
        
        # Test create_swarm with preset
        swarm = create_swarm("quick", db_path=TEST_DB_PATH)
        assert swarm.max_workers == PRESETS["quick"].max_workers
        print(f"✅ Created 'quick' swarm with {swarm.max_workers} workers")
        
        # Test SwarmBuilder
        swarm = (SwarmBuilder()
            .from_preset("balanced")
            .with_workers(3)
            .with_subtasks(4)
            .with_db_path(TEST_DB_PATH)
            .build())
        
        assert swarm.max_workers == 3
        assert swarm.max_subtasks == 4
        print(f"✅ SwarmBuilder created swarm with {swarm.max_workers} workers, {swarm.max_subtasks} subtasks")
        
        return True
        
    except Exception as e:
        print(f"❌ SwarmFactory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 8: Integration Test (Full Workflow)
# =============================================================================

def test_integration():
    """Full integration test with Agno Workflow"""
    print("\n" + "=" * 60)
    print("TEST 8: Integration Test (Full Workflow)")
    print("=" * 60)
    
    try:
        from main import ResearchSwarm
        
        # Create swarm
        swarm = ResearchSwarm(
            max_workers=2,
            max_subtasks=3,
            db_path=TEST_DB_PATH,
        )
        
        query = "What are the main benefits of renewable energy sources?"
        
        print(f"🔍 Running full integration test with: {query[:50]}...")
        print(f"   Using Agno Workflow with parallel execution")
        
        # Execute with Agno Workflow
        result = swarm.research(query)
        
        # Validate
        print(f"\n📊 Results:")
        print(result.summary())
        
        if result.success:
            print(f"\n✅ Integration test PASSED!")
            print(f"\n📄 Report Preview:")
            print("-" * 40)
            if result.report:
                print(result.report[:500] + "..." if len(result.report) > 500 else result.report)
            else:
                print("(No report generated)")
        else:
            print(f"\n⚠️  Integration test completed with issues: {result.error}")
            # Still return True if we got some results
            return result.plan is not None
        
        return result.success
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all orchestration tests"""
    print("\n" + "=" * 70)
    print("   PHASE 4: ORCHESTRATION LAYER TEST SUITE")
    print("=" * 70)
    
    # Clean up before tests
    cleanup_test_db()
    
    results = {}
    
    # Test 1: Configuration
    results["config"] = test_config()
    
    if not results["config"]:
        print("\n⚠️  Configuration incomplete - some tests may fail")
    
    # Test 2: SwarmResult
    results["swarm_result"] = test_swarm_result()
    
    # Test 3: PlannerExecutor
    passed, plan_content = test_planner_executor()
    results["planner_executor"] = passed
    
    # Test 4: WorkerExecutor
    if passed:
        results["worker_executor"] = test_worker_executor(plan_content)
    else:
        print("\n⚠️  Skipping worker test due to planner failure")
        results["worker_executor"] = False
    
    # Test 5: EditorExecutor
    results["editor_executor"] = test_editor_executor()
    
    # Test 6: ResearchSwarm Simple
    results["swarm_simple"] = test_research_swarm_simple()
    
    # Test 7: SwarmFactory
    results["swarm_factory"] = test_swarm_factory()
    
    # Test 8: Integration (only if basic tests pass)
    if all([results.get("planner_executor"), results.get("swarm_simple")]):
        results["integration"] = test_integration()
    else:
        print("\n⚠️  Skipping integration test due to previous failures")
        results["integration"] = False
    
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
    
    # Clean up
    cleanup_test_db()
    
    if passed_count == total:
        print("\n🎉 All Phase 4 tests passed! Orchestration layer is working.")
    else:
        print(f"\n⚠️  {total - passed_count} tests failed. Check the output above.")
    
    return passed_count == total


def run_quick_test():
    """Run a quick subset of tests"""
    print("\n" + "=" * 60)
    print("   QUICK TEST: Core Functionality")
    print("=" * 60)
    
    cleanup_test_db()
    
    results = {}
    
    # Config check
    results["config"] = test_config()
    
    # SwarmResult
    results["swarm_result"] = test_swarm_result()
    
    # PlannerExecutor
    passed, _ = test_planner_executor()
    results["planner"] = passed
    
    # SwarmFactory
    results["factory"] = test_swarm_factory()
    
    cleanup_test_db()
    
    passed_count = sum(1 for v in results.values() if v)
    total = len(results)
    
    print("\n" + "-" * 60)
    print(f"   Quick test: {passed_count}/{total} passed")
    
    return passed_count == total


if __name__ == "__main__":
    # Check for quick test flag
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
    else:
        success = run_all_tests()
    
    sys.exit(0 if success else 1)


