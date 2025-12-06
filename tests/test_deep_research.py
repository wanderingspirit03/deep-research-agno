"""
Test Suite for Deep Research System Upgrade

Tests the following without running full 30-minute research:
1. Configuration updates (timeouts, DeepResearchConfig)
2. New agent initialization (Critic, Domain Experts)
3. Schema validation
4. Retry utilities
5. DeepResearchSwarm initialization
6. Quick integration test with minimal iterations

Run with: python test_deep_research.py
Or: pytest test_deep_research.py -v
"""
import os
import sys
import json
import tempfile
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfiguration:
    """Test configuration updates"""
    
    def test_model_config_has_timeouts(self):
        """Verify timeout settings are present"""
        from config import ModelConfig
        
        cfg = ModelConfig()
        
        assert hasattr(cfg, 'planner_timeout')
        assert hasattr(cfg, 'worker_timeout')
        assert hasattr(cfg, 'editor_timeout')
        assert hasattr(cfg, 'critic_timeout')
        
        # Should be 1800 seconds (30 minutes)
        assert cfg.planner_timeout == 1800
        assert cfg.worker_timeout == 1800
        assert cfg.editor_timeout == 1800
        assert cfg.critic_timeout == 1800
        
        print("✅ Timeout settings configured correctly")
    
    def test_model_config_has_retry_settings(self):
        """Verify retry settings are present"""
        from config import ModelConfig
        
        cfg = ModelConfig()
        
        assert hasattr(cfg, 'max_retries')
        assert hasattr(cfg, 'retry_delay_base')
        assert cfg.max_retries == 3
        assert cfg.retry_delay_base == 5.0
        
        print("✅ Retry settings configured correctly")
    
    def test_deep_research_config_exists(self):
        """Verify DeepResearchConfig exists with proper defaults"""
        from config import DeepResearchConfig
        
        cfg = DeepResearchConfig()
        
        assert cfg.max_iterations == 3
        assert cfg.quality_threshold == 80
        assert cfg.checkpoint_enabled == True
        assert cfg.min_sources_per_subtask == 5
        assert cfg.target_findings_per_subtask == 10
        assert cfg.min_report_words == 3000
        
        print("✅ DeepResearchConfig configured correctly")
    
    def test_config_includes_deep_research(self):
        """Verify main Config includes deep_research"""
        from config import config
        
        assert hasattr(config, 'deep_research')
        assert config.deep_research.max_iterations == 3
        
        print("✅ Config includes deep_research settings")


# =============================================================================
# Test Schemas
# =============================================================================

class TestSchemas:
    """Test Pydantic schemas"""
    
    def test_deep_subtask_schema(self):
        """Test DeepSubtask creation and validation"""
        from agents.schemas import DeepSubtask, ResearchPhase, SearchType
        
        subtask = DeepSubtask(
            id=1,
            phase=ResearchPhase.FOUNDATION,
            focus="Background research",
            primary_query="What is machine learning?",
            alternative_queries=["ML basics", "machine learning introduction"],
            search_types=[SearchType.GENERAL, SearchType.ACADEMIC],
            expected_sources=10,
            dependencies=[],
        )
        
        assert subtask.id == 1
        assert subtask.phase == ResearchPhase.FOUNDATION
        assert len(subtask.alternative_queries) == 2
        assert SearchType.ACADEMIC in subtask.search_types
        
        print("✅ DeepSubtask schema works correctly")
    
    def test_quality_finding_schema(self):
        """Test QualityFinding with score calculation"""
        from agents.schemas import QualityFinding, SourceAuthority
        
        finding = QualityFinding(
            content="Important research finding",
            source_url="https://example.com/paper",
            source_title="Research Paper",
            quality_score=5,
            relevance_score=4,
            recency_score=3,
            authority_type=SourceAuthority.PEER_REVIEWED,
            key_statistics=["95% accuracy", "1M parameters"],
        )
        
        # Test overall score calculation
        expected_score = 5 * 0.4 + 4 * 0.4 + 3 * 0.2
        assert abs(finding.overall_score - expected_score) < 0.01
        
        print("✅ QualityFinding schema works correctly")
    
    def test_critic_evaluation_schema(self):
        """Test CriticEvaluation schema"""
        from agents.schemas import CriticEvaluation, GapAnalysis
        
        evaluation = CriticEvaluation(
            overall_score=75,
            coverage_score=80,
            source_quality_score=70,
            evidence_strength_score=75,
            balance_score=70,
            strengths=["Good coverage", "Multiple perspectives"],
            weaknesses=["Missing academic sources"],
            critical_gaps=[
                GapAnalysis(
                    gap_description="Need more recent data",
                    importance=4,
                    suggested_queries=["latest research 2024"],
                )
            ],
            follow_up_queries=["follow up query 1"],
            ready_for_synthesis=False,
        )
        
        assert evaluation.overall_score == 75
        assert len(evaluation.critical_gaps) == 1
        assert evaluation.critical_gaps[0].importance == 4
        
        print("✅ CriticEvaluation schema works correctly")
    
    def test_research_checkpoint_schema(self):
        """Test ResearchCheckpoint for serialization"""
        from agents.schemas import ResearchCheckpoint
        
        checkpoint = ResearchCheckpoint(
            checkpoint_id="test-123",
            session_id="session-456",
            phase="research",
            iteration=2,
            findings=[{"content": "test", "source": "test.com"}],
            can_resume=True,
        )
        
        # Test serialization
        data = checkpoint.model_dump()
        assert data["checkpoint_id"] == "test-123"
        assert data["iteration"] == 2
        
        print("✅ ResearchCheckpoint schema works correctly")


# =============================================================================
# Test Retry Utilities
# =============================================================================

class TestRetryUtils:
    """Test retry utilities"""
    
    def test_with_retry_decorator(self):
        """Test synchronous retry decorator"""
        from infrastructure.retry_utils import with_retry
        
        call_count = 0
        
        @with_retry(max_retries=3, base_delay=0.1)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Simulated failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
        
        print("✅ with_retry decorator works correctly")
    
    def test_calculate_backoff_delay(self):
        """Test backoff delay calculation"""
        from infrastructure.retry_utils import calculate_backoff_delay
        
        # Without jitter
        delay_0 = calculate_backoff_delay(0, base_delay=2.0, jitter=False)
        delay_1 = calculate_backoff_delay(1, base_delay=2.0, jitter=False)
        delay_2 = calculate_backoff_delay(2, base_delay=2.0, jitter=False)
        
        assert delay_0 == 2.0
        assert delay_1 == 4.0
        assert delay_2 == 8.0
        
        # Test max delay cap
        delay_large = calculate_backoff_delay(10, base_delay=2.0, max_delay=60.0, jitter=False)
        assert delay_large == 60.0
        
        print("✅ calculate_backoff_delay works correctly")
    
    def test_is_retriable_error(self):
        """Test retriable error detection"""
        from infrastructure.retry_utils import is_retriable_error
        
        assert is_retriable_error(Exception("Connection timeout"))
        assert is_retriable_error(Exception("Rate limit exceeded"))
        assert is_retriable_error(Exception("503 Service Unavailable"))
        assert not is_retriable_error(Exception("Invalid API key"))
        
        print("✅ is_retriable_error works correctly")


# =============================================================================
# Test New Agents
# =============================================================================

class TestNewAgents:
    """Test new agent initialization (without LLM calls)"""
    
    def test_critic_agent_initialization(self):
        """Test CriticAgent can be initialized"""
        from agents.critic import CriticAgent
        
        critic = CriticAgent(
            model_id="gpt-5-mini-2025-08-07",
            quality_threshold=80,
        )
        
        assert critic.model_id == "gpt-5-mini-2025-08-07"
        assert critic.quality_threshold == 80
        assert critic.temperature == 0.2  # Low for consistency
        
        print("✅ CriticAgent initializes correctly")
    
    def test_critic_quick_assess(self):
        """Test CriticAgent quick_assess without LLM"""
        from agents.critic import CriticAgent
        
        critic = CriticAgent()
        
        findings = [
            {"content": "Finding 1", "source_url": "https://a.com", "search_type": "general"},
            {"content": "Finding 2", "source_url": "https://b.com", "search_type": "academic"},
            {"content": "Finding 3", "source_url": "https://c.com", "search_type": "general"},
            {"content": "Finding 4", "source_url": "https://b.com", "search_type": "academic"},  # Same source
        ]
        
        assessment = critic.quick_assess(findings)
        
        assert assessment["num_findings"] == 4
        assert assessment["num_sources"] == 3  # 3 unique sources
        assert assessment["academic_ratio"] == 0.5  # 2/4 academic
        assert "estimated_score" in assessment
        assert "needs_more_research" in assessment
        
        print("✅ CriticAgent quick_assess works correctly")
    
    def test_domain_expert_configs(self):
        """Test domain expert configurations"""
        from agents.domain_experts import EXPERT_CONFIGS, list_expert_types
        
        assert "technical" in EXPERT_CONFIGS
        assert "industry" in EXPERT_CONFIGS
        assert "skeptic" in EXPERT_CONFIGS
        assert "futurist" in EXPERT_CONFIGS
        assert "academic" in EXPERT_CONFIGS
        
        experts = list_expert_types()
        assert len(experts) == 5
        
        print("✅ Domain expert configs loaded correctly")
    
    def test_create_expert_agent(self):
        """Test expert agent creation"""
        from agents.domain_experts import create_expert_agent
        
        expert = create_expert_agent("technical")
        
        assert expert.expert_type == "technical"
        assert expert.config["name"] == "Technical Expert"
        assert "Algorithm design" in expert.config["focus_areas"][0]
        
        print("✅ create_expert_agent works correctly")
    
    def test_create_expert_panel(self):
        """Test expert panel creation"""
        from agents.domain_experts import create_expert_panel
        
        panel = create_expert_panel(["technical", "industry"])
        
        assert len(panel) == 2
        assert panel[0].expert_type == "technical"
        assert panel[1].expert_type == "industry"
        
        print("✅ create_expert_panel works correctly")


# =============================================================================
# Test Swarm Factory Updates
# =============================================================================

class TestSwarmFactory:
    """Test swarm factory updates"""
    
    def test_deep_research_preset_exists(self):
        """Test deep_research preset is available"""
        from swarm_factory import PRESETS
        
        assert "deep_research" in PRESETS
        assert "express_deep" in PRESETS
        
        deep = PRESETS["deep_research"]
        assert deep.max_iterations == 3
        assert deep.quality_threshold == 80
        assert deep.use_experts == True
        assert deep.max_subtasks == 15
        
        express = PRESETS["express_deep"]
        assert express.max_iterations == 1
        assert express.use_experts == False
        
        print("✅ Deep research presets configured correctly")
    
    def test_list_presets(self):
        """Test listing all presets"""
        from swarm_factory import list_presets
        
        presets = list_presets()
        
        assert "quick" in presets
        assert "balanced" in presets
        assert "deep" in presets
        assert "deep_research" in presets
        assert "express_deep" in presets
        
        print("✅ list_presets includes all presets")


# =============================================================================
# Test DeepResearchSwarm
# =============================================================================

class TestDeepResearchSwarm:
    """Test DeepResearchSwarm class"""
    
    def test_initialization(self):
        """Test DeepResearchSwarm initialization"""
        from main import DeepResearchSwarm
        
        with tempfile.TemporaryDirectory() as tmpdir:
            swarm = DeepResearchSwarm(
                max_workers=3,
                max_subtasks=5,
                max_iterations=2,
                quality_threshold=70,
                checkpoint_dir=f"{tmpdir}/checkpoints",
                db_path=f"{tmpdir}/test_kb",
            )
            
            assert swarm.max_workers == 3
            assert swarm.max_subtasks == 5
            assert swarm.max_iterations == 2
            assert swarm.quality_threshold == 70
            
            print("✅ DeepResearchSwarm initializes correctly")
    
    def test_lazy_agent_initialization(self):
        """Test that agents are lazily initialized"""
        from main import DeepResearchSwarm
        
        with tempfile.TemporaryDirectory() as tmpdir:
            swarm = DeepResearchSwarm(
                checkpoint_dir=f"{tmpdir}/checkpoints",
                db_path=f"{tmpdir}/test_kb",
            )
            
            # Agents should not be initialized yet
            assert swarm._planner is None
            assert swarm._worker is None
            assert swarm._editor is None
            assert swarm._critic is None
            
            print("✅ Agents are lazily initialized")
    
    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading"""
        from main import DeepResearchSwarm
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = f"{tmpdir}/checkpoints"
            
            swarm = DeepResearchSwarm(
                checkpoint_dir=checkpoint_dir,
                db_path=f"{tmpdir}/test_kb",
            )
            
            # Save a checkpoint
            test_data = {"test_key": "test_value", "count": 42}
            swarm._save_checkpoint("test_phase", test_data)
            
            # Load the checkpoint
            loaded = swarm._load_checkpoint("test_phase")
            
            assert loaded is not None
            assert loaded["data"]["test_key"] == "test_value"
            assert loaded["data"]["count"] == 42
            assert loaded["phase"] == "test_phase"
            
            print("✅ Checkpoint save/load works correctly")


# =============================================================================
# Test Agent Exports
# =============================================================================

class TestAgentExports:
    """Test that all new agents are properly exported"""
    
    def test_all_exports_available(self):
        """Test all new exports are available from agents module"""
        from agents import (
            # Core agents
            PlannerAgent,
            WorkerAgent,
            EditorAgent,
            # New agents
            CriticAgent,
            DomainExpertAgent,
            create_expert_agent,
            create_expert_panel,
            # Schemas
            ResearchPhase,
            SearchType,
            SourceAuthority,
            DeepSubtask,
            QualityFinding,
            CriticEvaluation,
            ResearchIteration,
            ResearchCheckpoint,
        )
        
        # Just verify they're importable
        assert PlannerAgent is not None
        assert CriticAgent is not None
        assert DomainExpertAgent is not None
        assert ResearchPhase is not None
        assert CriticEvaluation is not None
        
        print("✅ All exports are available")


# =============================================================================
# Integration Test (Quick)
# =============================================================================

class TestQuickIntegration:
    """Quick integration test without full research"""
    
    @pytest.mark.skipif(
        not os.getenv("LITELLM_API_KEY"),
        reason="LITELLM_API_KEY not set"
    )
    def test_planner_creates_enhanced_plan(self):
        """Test planner creates plan with enhanced prompts"""
        from agents.planner import PlannerAgent
        
        planner = PlannerAgent(max_subtasks=5)
        
        # Simple query to test planning
        plan = planner.plan("What is machine learning?")
        
        assert plan is not None
        assert len(plan.subtasks) > 0
        assert len(plan.subtasks) <= 5
        
        # Check plan has proper structure
        assert plan.original_query == "What is machine learning?"
        assert plan.summary != ""
        
        print(f"✅ Planner created plan with {len(plan.subtasks)} subtasks")
        print(f"   Summary: {plan.summary[:100]}...")


# =============================================================================
# Main Test Runner
# =============================================================================

def run_all_tests():
    """Run all tests without pytest"""
    print("\n" + "=" * 70)
    print("  DEEP RESEARCH SYSTEM - TEST SUITE")
    print("=" * 70 + "\n")
    
    test_classes = [
        TestConfiguration,
        TestSchemas,
        TestRetryUtils,
        TestNewAgents,
        TestSwarmFactory,
        TestDeepResearchSwarm,
        TestAgentExports,
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test_class in test_classes:
        print(f"\n📋 {test_class.__name__}")
        print("-" * 50)
        
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except Exception as e:
                    failed += 1
                    errors.append((f"{test_class.__name__}.{method_name}", str(e)))
                    print(f"❌ {method_name}: {e}")
    
    print("\n" + "=" * 70)
    print(f"  RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if errors:
        print("\n❌ Failed tests:")
        for name, error in errors:
            print(f"   - {name}: {error}")
    
    return failed == 0


if __name__ == "__main__":
    import sys
    
    # Check if running with pytest
    if "pytest" in sys.modules:
        # pytest will handle test discovery
        pass
    else:
        # Run manually
        success = run_all_tests()
        sys.exit(0 if success else 1)


