#!/usr/bin/env python3
"""
Full integration test for the Editor synthesis flow.

This test:
1. Uses the existing research_kb with 82 findings
2. Generates findings index
3. Calls the Editor to synthesize a comprehensive report
4. Verifies the report quality
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from dotenv import load_dotenv
load_dotenv()


def test_full_editor_synthesis():
    """Test the complete Editor synthesis flow"""
    
    print("=" * 70)
    print("  FULL EDITOR SYNTHESIS TEST")
    print("=" * 70)
    print()
    
    # Check for existing KB
    kb_path = "./research_kb"
    if not os.path.exists(kb_path):
        print("❌ No existing research_kb found. Run research first.")
        return False
    
    from infrastructure.knowledge_tools import KnowledgeTools
    from agents.editor import EditorAgent
    
    # Initialize
    print("1. Initializing components...")
    kt = KnowledgeTools(db_path=kb_path)
    editor = EditorAgent(knowledge_tools=kt)
    
    # Generate index
    print("\n2. Generating findings index...")
    print("-" * 50)
    index = kt.get_findings_index()
    print(f"   Index length: {len(index)} chars")
    
    # Show index preview
    print("\n   Index preview (first 30 lines):")
    for line in index.split("\n")[:30]:
        print(f"   {line}")
    print("   ...")
    
    # Test search functionality
    print("\n3. Testing search_knowledge tool...")
    print("-" * 50)
    
    test_queries = [
        "AI agent architecture components",
        "multi-agent coordination collaboration",
        "reasoning planning capabilities",
    ]
    
    for query in test_queries:
        results = kt.search_knowledge(query, top_k=3)
        result_count = results.count("###")  # Count result headers
        content_length = len(results)
        print(f"   '{query}': {result_count} results, {content_length} chars")
    
    # Call Editor synthesis
    print("\n4. Calling Editor synthesis...")
    print("-" * 50)
    print("   This may take 2-5 minutes as the Editor searches and writes...")
    print()
    
    query = "What are the key breakthroughs in AI agents and multi-agent systems in 2024?"
    
    try:
        report = editor.synthesize(query, index)
        
        print("\n5. Analyzing report quality...")
        print("-" * 50)
        
        # Calculate metrics
        word_count = len(report.split())
        char_count = len(report)
        line_count = len(report.split("\n"))
        
        # Check for key sections
        sections = {
            "Abstract": "abstract" in report.lower(),
            "Introduction": "introduction" in report.lower(),
            "Background": "background" in report.lower() or "definition" in report.lower(),
            "Literature Review": "literature" in report.lower() or "state of the art" in report.lower(),
            "Critical Analysis": "critical" in report.lower() or "analysis" in report.lower(),
            "Future": "future" in report.lower(),
            "Conclusion": "conclusion" in report.lower(),
            "References": "reference" in report.lower() or "source" in report.lower(),
        }
        
        # Check for citations
        citation_count = report.count("[")
        
        print(f"   Word count: {word_count}")
        print(f"   Character count: {char_count}")
        print(f"   Line count: {line_count}")
        print(f"   Citation count: ~{citation_count}")
        print()
        print("   Section checks:")
        for section, found in sections.items():
            icon = "✅" if found else "❌"
            print(f"   {icon} {section}")
        
        # Quality assessment
        print()
        print("6. Quality Assessment")
        print("-" * 50)
        
        issues = []
        if word_count < 3000:
            issues.append(f"Word count too low: {word_count} (target: 5000+)")
        if citation_count < 10:
            issues.append(f"Too few citations: {citation_count} (target: 20+)")
        if not sections["References"]:
            issues.append("Missing references section")
        
        sections_found = sum(1 for v in sections.values() if v)
        if sections_found < 5:
            issues.append(f"Only {sections_found}/8 sections found")
        
        if issues:
            print("   ⚠️  Issues found:")
            for issue in issues:
                print(f"      - {issue}")
        else:
            print("   ✅ Report meets quality standards!")
        
        # Save report
        output_file = "test_editor_report.md"
        with open(output_file, "w") as f:
            f.write(f"# Test Editor Report\n\n")
            f.write(f"**Query:** {query}\n\n")
            f.write(f"**Word Count:** {word_count}\n\n")
            f.write(f"**Generated with:** Tool-based discovery\n\n")
            f.write("---\n\n")
            f.write(report)
        
        print(f"\n   Report saved to: {output_file}")
        
        # Show report preview
        print("\n7. Report Preview (first 2000 chars)")
        print("-" * 50)
        print(report[:2000])
        print("\n... (truncated)")
        
        return word_count >= 2000 and sections_found >= 4
        
    except Exception as e:
        print(f"\n❌ Editor synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    success = test_full_editor_synthesis()
    
    print()
    print("=" * 70)
    if success:
        print("  ✅ FULL EDITOR TEST PASSED")
    else:
        print("  ❌ FULL EDITOR TEST FAILED")
    print("=" * 70)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

