#!/usr/bin/env python3
"""Quick express research test script"""
import os
import sys

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

from dotenv import load_dotenv
load_dotenv()

print('=' * 70)
print('  RUNNING EXPRESS DEEP RESEARCH')
print('=' * 70)
print()

from swarm_factory import deep_research

query = 'What are the key breakthroughs in AI agents and multi-agent systems in 2024?'
print(f'Query: {query}')
print()
print('Starting research (this may take 5-10 minutes)...')
print()

result = deep_research(query, express=True)

print()
print('=' * 70)
print('  RESEARCH COMPLETE')
print('=' * 70)
print()
print(result.summary())
print()
print('-' * 70)
print('  REPORT PREVIEW (first 3000 chars)')
print('-' * 70)
print()
print(result.report[:3000] if len(result.report) > 3000 else result.report)

if len(result.report) > 3000:
    print()
    print('... (report truncated for display)')

# Save full report
with open('express_research_report.md', 'w') as f:
    f.write(f'# Express Research Report\n\n')
    f.write(f'**Query:** {query}\n\n')
    f.write(f'**Success:** {result.success}\n\n')
    f.write('---\n\n')
    f.write(result.report)

print()
print('=' * 70)
print('Full report saved to: express_research_report.md')
print('=' * 70)



