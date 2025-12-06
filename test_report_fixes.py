#!/usr/bin/env python3
"""
Tests for the report generation fixes:
1. _get_all_findings() - proper structured data from LanceDB
2. _generate_fallback_report() - comprehensive output
3. _build_synthesis_context() - includes findings
"""
import os
import sys
import tempfile
import shutil

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from dotenv import load_dotenv
load_dotenv()

import unittest
from unittest.mock import MagicMock, patch


class TestGetAllFindings(unittest.TestCase):
    """Test the _get_all_findings method returns structured data"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_kb")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_get_all_findings_returns_list(self):
        """Test that _get_all_findings returns a list"""
        from main import DeepResearchSwarm
        
        swarm = DeepResearchSwarm(
            max_workers=2,
            max_subtasks=3,
            db_path=self.test_db_path,
        )
        
        findings = swarm._get_all_findings()
        self.assertIsInstance(findings, list)
        print(f"✅ _get_all_findings returns list: {type(findings)}")
    
    def test_get_all_findings_structured_data(self):
        """Test that findings have proper structure when data exists"""
        from main import DeepResearchSwarm
        from infrastructure.knowledge_tools import KnowledgeTools
        
        # Create swarm with test DB
        swarm = DeepResearchSwarm(
            max_workers=2,
            max_subtasks=3,
            db_path=self.test_db_path,
        )
        
        # Add a test finding
        swarm.knowledge_tools.save_finding(
            content="Test finding content about AI agents and their capabilities in 2024",
            source_url="https://example.com/test",
            source_title="Test Source Title",
            search_type="academic",
            verified=True,
            subtask_id=1,
            worker_id="test_worker",
        )
        
        # Get findings
        findings = swarm._get_all_findings()
        
        # Verify structure
        self.assertGreater(len(findings), 0, "Should have at least one finding")
        
        finding = findings[0]
        print(f"\n📋 Finding structure: {list(finding.keys())}")
        
        # Check required fields exist
        required_fields = ["id", "content", "source_url", "source_title", "search_type", "verified"]
        for field in required_fields:
            self.assertIn(field, finding, f"Finding should have '{field}' field")
        
        # Verify content is actual content, not markdown
        self.assertNotIn("##", finding["content"], "Content should not contain markdown headers")
        self.assertNotIn("**Query:**", finding["content"], "Content should not contain query markers")
        
        print(f"✅ Finding has all required fields")
        print(f"   - id: {finding['id']}")
        print(f"   - content: {finding['content'][:50]}...")
        print(f"   - source_url: {finding['source_url']}")
        print(f"   - source_title: {finding['source_title']}")
        print(f"   - search_type: {finding['search_type']}")
        print(f"   - verified: {finding['verified']}")


class TestGenerateFallbackReport(unittest.TestCase):
    """Test the _generate_fallback_report method produces comprehensive output"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_kb")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_fallback_report_with_empty_findings(self):
        """Test fallback report handles empty findings"""
        from main import DeepResearchSwarm
        
        swarm = DeepResearchSwarm(
            max_workers=2,
            max_subtasks=3,
            db_path=self.test_db_path,
        )
        
        report = swarm._generate_fallback_report("Test query", [])
        
        self.assertIn("# Research Report:", report)
        self.assertIn("0 research findings", report)
        print(f"✅ Fallback report handles empty findings")
        print(f"   Report length: {len(report)} chars")
    
    def test_fallback_report_with_structured_findings(self):
        """Test fallback report properly uses structured findings"""
        from main import DeepResearchSwarm
        
        swarm = DeepResearchSwarm(
            max_workers=2,
            max_subtasks=3,
            db_path=self.test_db_path,
        )
        
        # Create test findings with full structure
        test_findings = [
            {
                "id": "abc123",
                "content": "GPT-4 achieved 86.4% on MMLU benchmark, demonstrating significant improvements in reasoning capabilities.",
                "source_url": "https://arxiv.org/abs/2303.08774",
                "source_title": "GPT-4 Technical Report",
                "search_type": "academic",
                "verified": True,
                "subtask_id": 1,
                "worker_id": "W01",
            },
            {
                "id": "def456",
                "content": "Multi-agent systems showed 40% improvement in complex task completion when using specialized role assignment.",
                "source_url": "https://example.com/multi-agent",
                "source_title": "Multi-Agent Coordination Study",
                "search_type": "academic",
                "verified": True,
                "subtask_id": 2,
                "worker_id": "W02",
            },
            {
                "id": "ghi789",
                "content": "AutoGPT and similar autonomous agents gained significant traction in 2024 with improved planning capabilities.",
                "source_url": "https://blog.example.com/autogpt",
                "source_title": "AutoGPT Blog Post",
                "search_type": "general",
                "verified": False,
                "subtask_id": 3,
                "worker_id": "W03",
            },
        ]
        
        report = swarm._generate_fallback_report("AI agents breakthroughs 2024", test_findings)
        
        # Check report structure
        self.assertIn("# Research Report:", report)
        self.assertIn("Executive Summary", report)
        self.assertIn("Key Findings", report)
        self.assertIn("Sources", report)
        
        # Check findings count
        self.assertIn("3 research findings", report)
        self.assertIn("3 unique sources", report)  # 3 different URLs = 3 sources
        
        # Check academic/general separation
        self.assertIn("Academic", report)
        self.assertIn("General", report)
        
        # Check source titles appear
        self.assertIn("GPT-4 Technical Report", report)
        self.assertIn("Multi-Agent Coordination Study", report)
        
        # Check verification icons
        self.assertIn("✅", report)
        self.assertIn("❌", report)
        
        # Check content is included (not just truncated to 200 chars)
        self.assertIn("86.4% on MMLU", report)
        self.assertIn("40% improvement", report)
        
        print(f"✅ Fallback report is comprehensive")
        print(f"   Report length: {len(report)} chars")
        print(f"   Contains academic section: {'Academic' in report}")
        print(f"   Contains general section: {'General' in report}")
        print(f"   Contains sources: {'Sources' in report}")
        
    def test_fallback_report_length_is_reasonable(self):
        """Test that fallback report is longer than the old broken version"""
        from main import DeepResearchSwarm
        
        swarm = DeepResearchSwarm(
            max_workers=2,
            max_subtasks=3,
            db_path=self.test_db_path,
        )
        
        # Create 10 test findings
        test_findings = [
            {
                "id": f"id{i}",
                "content": f"This is finding number {i} with substantial content about AI breakthroughs. " * 5,
                "source_url": f"https://example.com/source{i}",
                "source_title": f"Source Title {i}",
                "search_type": "academic" if i % 2 == 0 else "general",
                "verified": i % 3 == 0,
                "subtask_id": i,
                "worker_id": f"W{i:02d}",
            }
            for i in range(10)
        ]
        
        report = swarm._generate_fallback_report("Test query", test_findings)
        
        # Old broken version produced ~500-1000 chars
        # New version should produce significantly more
        self.assertGreater(len(report), 2000, "Report should be at least 2000 chars")
        
        print(f"✅ Fallback report length is reasonable: {len(report)} chars")


class TestBuildSynthesisContext(unittest.TestCase):
    """Test the _build_synthesis_context method includes findings"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_kb")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_synthesis_context_includes_findings(self):
        """Test that synthesis context includes findings data"""
        from main import DeepResearchSwarm
        
        swarm = DeepResearchSwarm(
            max_workers=2,
            max_subtasks=3,
            db_path=self.test_db_path,
        )
        
        # Set up test findings
        swarm.all_findings = [
            {
                "id": "test1",
                "content": "Important finding about transformer architectures",
                "source_url": "https://arxiv.org/test1",
                "source_title": "Transformer Paper",
                "search_type": "academic",
                "verified": True,
            },
            {
                "id": "test2",
                "content": "Another finding about attention mechanisms",
                "source_url": "https://arxiv.org/test2",
                "source_title": "Attention Paper",
                "search_type": "academic",
                "verified": False,
            },
        ]
        
        context = swarm._build_synthesis_context([])
        
        self.assertIsNotNone(context)
        self.assertIn("Research Findings Summary", context)
        self.assertIn("Total findings:", context)
        self.assertIn("Key Research Content", context)
        self.assertIn("transformer architectures", context)
        self.assertIn("attention mechanisms", context)
        
        print(f"✅ Synthesis context includes findings")
        print(f"   Context length: {len(context)} chars")
        print(f"   Contains findings summary: {'Research Findings Summary' in context}")
        print(f"   Contains actual content: {'transformer architectures' in context}")
    
    def test_synthesis_context_includes_expert_insights(self):
        """Test that synthesis context includes expert insights when provided"""
        from main import DeepResearchSwarm
        
        swarm = DeepResearchSwarm(
            max_workers=2,
            max_subtasks=3,
            db_path=self.test_db_path,
        )
        
        swarm.all_findings = []  # No findings
        
        expert_insights = [
            {
                "expert": "technical",
                "summary": "Technical analysis of the research",
                "insights": ["Insight 1", "Insight 2"],
                "concerns": ["Concern 1"],
            }
        ]
        
        context = swarm._build_synthesis_context(expert_insights)
        
        self.assertIsNotNone(context)
        self.assertIn("Expert Perspectives", context)
        self.assertIn("Technical Perspective", context)
        self.assertIn("Technical analysis", context)
        
        print(f"✅ Synthesis context includes expert insights")


class TestIntegration(unittest.TestCase):
    """Integration tests for the full flow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_kb")
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_findings_to_report_flow(self):
        """Test the complete flow from saving findings to generating report"""
        from main import DeepResearchSwarm
        
        swarm = DeepResearchSwarm(
            max_workers=2,
            max_subtasks=3,
            db_path=self.test_db_path,
        )
        
        # Save test findings to knowledge base
        test_findings_data = [
            ("GPT-4 achieves state-of-the-art on multiple benchmarks", "https://openai.com/gpt4", "GPT-4 Report", "academic", True),
            ("Claude 3 introduces improved reasoning capabilities", "https://anthropic.com/claude3", "Claude 3 Announcement", "general", True),
            ("LLaMA 3 open-source model released by Meta", "https://meta.ai/llama3", "LLaMA 3 Release", "general", False),
        ]
        
        for content, url, title, stype, verified in test_findings_data:
            swarm.knowledge_tools.save_finding(
                content=content,
                source_url=url,
                source_title=title,
                search_type=stype,
                verified=verified,
                subtask_id=1,
                worker_id="test",
            )
        
        # Get findings through the fixed method
        findings = swarm._get_all_findings()
        
        # Verify we got structured data
        self.assertEqual(len(findings), 3)
        for f in findings:
            self.assertIn("content", f)
            self.assertIn("source_url", f)
            self.assertNotIn("##", f["content"])  # No markdown artifacts
        
        # Generate fallback report
        report = swarm._generate_fallback_report("LLM breakthroughs 2024", findings)
        
        # Verify report quality (1000+ chars for 3 findings is reasonable)
        self.assertGreater(len(report), 1000)
        self.assertIn("GPT-4", report)
        self.assertIn("Claude 3", report)
        self.assertIn("LLaMA 3", report)
        self.assertIn("openai.com", report)
        
        print(f"✅ Full flow works correctly")
        print(f"   Findings retrieved: {len(findings)}")
        print(f"   Report length: {len(report)} chars")
        print(f"   All sources included: {all(url in report for _, url, _, _, _ in test_findings_data)}")


def run_tests():
    """Run all tests with verbose output"""
    print("=" * 70)
    print("  TESTING REPORT GENERATION FIXES")
    print("=" * 70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestGetAllFindings))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateFallbackReport))
    suite.addTests(loader.loadTestsFromTestCase(TestBuildSynthesisContext))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 70)
    if result.wasSuccessful():
        print("  ✅ ALL TESTS PASSED!")
    else:
        print(f"  ❌ TESTS FAILED: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

