"""
Regenerate Report from Existing Knowledge Base

This script regenerates a research report from findings already stored
in the LanceDB knowledge base, without re-running the entire research.

Useful for:
- Testing improved report formats
- Re-synthesizing with different parameters
- Recovering from editor timeouts
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Configure LiteLLM
import litellm
litellm.drop_params = True

from infrastructure.knowledge_tools import KnowledgeTools
from config import config


def get_all_findings(kb: KnowledgeTools) -> list:
    """Retrieve all findings from knowledge base"""
    try:
        df = kb.table.to_pandas()
        df = df[df["id"] != "init"]
        
        findings = []
        for _, row in df.iterrows():
            findings.append({
                "id": row.get("id", ""),
                "content": row.get("content", ""),
                "source_url": row.get("source_url", ""),
                "source_title": row.get("source_title", ""),
                "search_type": row.get("search_type", "general"),
                "verified": row.get("verified", False),
                "subtask_id": row.get("subtask_id", 0),
                "worker_id": row.get("worker_id", ""),
                "timestamp": row.get("timestamp", ""),
            })
        
        return findings
    except Exception as e:
        print(f"Error retrieving findings: {e}")
        return []


def regenerate_fallback_report(query: str, findings: list) -> str:
    """Generate report using the improved fallback generator"""
    from main import DeepResearchSwarm
    
    # Create a swarm instance just to use the fallback report generator
    swarm = DeepResearchSwarm(db_path=config.knowledge.db_path)
    
    return swarm._generate_fallback_report(query, findings)


def regenerate_with_editor(query: str, kb: KnowledgeTools) -> str:
    """Attempt to regenerate using the full editor agent"""
    from agents.editor import EditorAgent
    
    editor = EditorAgent(
        model_id=config.models.editor,
        temperature=config.models.editor_temperature,
        knowledge_tools=kb,
    )
    
    print("Generating findings index...")
    findings_index = kb.get_findings_index()
    
    print("Running editor synthesis...")
    return editor.synthesize(query, findings_index)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Regenerate research report from existing knowledge base"
    )
    parser.add_argument(
        "query",
        help="The original research query"
    )
    parser.add_argument(
        "--db-path",
        default=config.knowledge.db_path,
        help=f"Path to LanceDB database (default: {config.knowledge.db_path})"
    )
    parser.add_argument(
        "--output", "-o",
        default="regenerated_report.md",
        help="Output file path (default: regenerated_report.md)"
    )
    parser.add_argument(
        "--mode",
        choices=["fallback", "editor"],
        default="fallback",
        help="Generation mode: 'fallback' (fast, structured) or 'editor' (full LLM synthesis)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("  REPORT REGENERATION")
    print("=" * 60)
    print(f"\nQuery: {args.query}")
    print(f"Database: {args.db_path}")
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    
    # Initialize knowledge tools
    kb = KnowledgeTools(db_path=args.db_path)
    
    # Get findings
    print("\nRetrieving findings from knowledge base...")
    findings = get_all_findings(kb)
    print(f"Found {len(findings)} findings")
    
    if len(findings) == 0:
        print("❌ No findings found in knowledge base!")
        return 1
    
    # Generate report
    print(f"\nGenerating report using {args.mode} mode...")
    
    if args.mode == "fallback":
        report = regenerate_fallback_report(args.query, findings)
    else:
        try:
            report = regenerate_with_editor(args.query, kb)
        except Exception as e:
            print(f"⚠️ Editor failed: {e}")
            print("Falling back to structured report...")
            report = regenerate_fallback_report(args.query, findings)
    
    # Save report
    with open(args.output, "w") as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {args.output}")
    print(f"   Length: {len(report)} characters (~{len(report.split())} words)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

