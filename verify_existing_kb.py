#!/usr/bin/env python3
"""
Quick verification of the fixes against the existing knowledge base
"""
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from dotenv import load_dotenv
load_dotenv()

print("=" * 70)
print("  VERIFYING FIXES WITH EXISTING KNOWLEDGE BASE")
print("=" * 70)
print()

from main import DeepResearchSwarm

# Use the existing research KB
swarm = DeepResearchSwarm(
    max_workers=5,
    max_subtasks=7,
    db_path="./research_kb",
)

print("1. Testing _get_all_findings()...")
print("-" * 50)

findings = swarm._get_all_findings()

print(f"   Total findings retrieved: {len(findings)}")

if findings:
    # Analyze findings
    academic = sum(1 for f in findings if f.get("search_type") == "academic")
    general = len(findings) - academic
    verified = sum(1 for f in findings if f.get("verified", False))
    
    print(f"   - Academic: {academic}")
    print(f"   - General: {general}")
    print(f"   - Verified: {verified}")
    
    # Check structure of first finding
    first = findings[0]
    print(f"\n   Sample finding structure:")
    print(f"   - id: {first.get('id', 'MISSING')}")
    print(f"   - content (first 100 chars): {first.get('content', '')[:100]}...")
    print(f"   - source_url: {first.get('source_url', 'MISSING')}")
    print(f"   - source_title: {first.get('source_title', 'MISSING')}")
    print(f"   - search_type: {first.get('search_type', 'MISSING')}")
    print(f"   - verified: {first.get('verified', 'MISSING')}")
    
    # Check for markdown artifacts (should NOT be present)
    has_markdown_artifacts = any(
        "##" in f.get("content", "") or "**Query:**" in f.get("content", "")
        for f in findings[:10]
    )
    if has_markdown_artifacts:
        print(f"\n   ⚠️  WARNING: Some findings contain markdown artifacts!")
    else:
        print(f"\n   ✅ No markdown artifacts in findings content")

print()
print("2. Testing _generate_fallback_report()...")
print("-" * 50)

query = "What are the key breakthroughs in AI agents and multi-agent systems in 2024?"
report = swarm._generate_fallback_report(query, findings)

print(f"   Report length: {len(report)} characters")
print(f"   Report lines: {len(report.splitlines())}")

# Check report structure
checks = [
    ("Has title", "# Research Report:" in report),
    ("Has executive summary", "Executive Summary" in report),
    ("Has academic section", "Academic" in report),
    ("Has general section", "General" in report),
    ("Has sources section", "## Sources" in report),
    ("Has verification icons", "✅" in report or "❌" in report),
    ("Has methodology note", "Methodology Note" in report),
]

print(f"\n   Report structure checks:")
for name, passed in checks:
    icon = "✅" if passed else "❌"
    print(f"   {icon} {name}")

print()
print("3. Saving sample report to 'fixed_fallback_report.md'...")
print("-" * 50)

with open("fixed_fallback_report.md", "w") as f:
    f.write(report)

print(f"   Saved {len(report)} chars to fixed_fallback_report.md")

print()
print("=" * 70)
print("  VERIFICATION COMPLETE")
print("=" * 70)
print()
print("Compare:")
print("  - OLD: full_research_report.md (short, broken)")
print("  - NEW: fixed_fallback_report.md (comprehensive)")



