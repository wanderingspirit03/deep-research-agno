# Isara Deep Research System

A PhD-level autonomous research system powered by multi-agent orchestration. The system conducts comprehensive research on any topic, synthesizing information from multiple sources into detailed, well-cited reports.

## Overview

The Isara Deep Research System uses a swarm of specialized AI agents to:
- **Plan** research strategies and decompose complex queries into subtasks
- **Search** multiple sources using Perplexity AI for comprehensive coverage
- **Analyze** findings with domain-specific expert agents
- **Critique** research quality and identify gaps
- **Synthesize** findings into comprehensive, publication-ready reports

### Key Features

- **Multi-Agent Architecture**: Planner, Worker, Critic, and Editor agents collaborate
- **Parallel Execution**: Multiple research workers operate concurrently
- **Quality-Driven Iteration**: Research loops until quality thresholds are met
- **Knowledge Persistence**: LanceDB vector store for findings and citations
- **Human-in-the-Loop (HITL)**: Optional Slack integration for human review
- **Observability**: Full tracing via Laminar (LMNR)
- **Modern Web UI**: Next.js frontend with real-time status updates

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PLANNER AGENT                               │
│  - Analyzes query complexity                                     │
│  - Creates research plan with subtasks                          │
│  - Identifies required domains and sources                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    WORKER AGENTS (Parallel)                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Worker 1 │ │Worker 2 │ │Worker 3 │ │Worker 4 │ │Worker 5 │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
│       │           │           │           │           │         │
│       ▼           ▼           ▼           ▼           ▼         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Perplexity Search + Parallel API            │   │
│  │              (URL extraction, academic sources)          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     KNOWLEDGE BASE (LanceDB)                     │
│  - Vector embeddings for semantic search                        │
│  - Source citations and metadata                                │
│  - Cross-subtask deduplication                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CRITIC AGENT                                │
│  - Evaluates research quality (0-100)                           │
│  - Identifies gaps and missing coverage                         │
│  - Optional: Routes to HITL for human review                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                   ┌──────────┴──────────┐
                   │ Quality < Threshold │
                   │   Loop back to      │
                   │   Workers with gaps │
                   └─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      EDITOR AGENT                                │
│  - Synthesizes all findings                                     │
│  - Generates structured report                                  │
│  - Includes methodology and limitations                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Final Research Report                         │
│  - Executive summary                                            │
│  - Detailed findings by section                                 │
│  - Full citations and references                                │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for frontend)
- **API Keys** (see Configuration section)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/isara-deep-research.git
cd isara-deep-research
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Set Up Frontend

```bash
cd frontend
npm install
cd ..
```

### 4. Configure Environment

```bash
# Copy example environment file
cp env.example.txt .env

# Edit .env with your API keys
nano .env  # or use your preferred editor
```

## Configuration

Create a `.env` file in the root directory with the following variables:

### Required API Keys

```bash
# Perplexity Search API (required)
PERPLEXITY_API_KEY=pplx-...

# LLM Provider - Choose one option:

# Option A: Direct API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Option B: LiteLLM Proxy (recommended for production)
LITELLM_API_BASE=https://your-proxy-url
LITELLM_API_KEY=your-litellm-key

# Parallel API for URL content extraction
PARALLEL_API_KEY=your-parallel-api-key
```

### Optional Configuration

```bash
# Daytona Sandbox (for code execution)
DAYTONA_API_KEY=your-daytona-api-key
DAYTONA_TARGET=us  # or 'eu'

# Human-in-the-Loop (Slack integration)
HITL_ENABLED=false
SLACK_TOKEN=xoxb-your-slack-bot-token
HITL_DEFAULT_CHANNEL=C0123456789

# Observability (Laminar)
LMNR_PROJECT_API_KEY=your-laminar-key

# Knowledge Base path
LANCEDB_PATH=./research_kb
```

### Validate Configuration

```bash
python config.py
```

## Running the System

### Quick Start (Both Servers)

```bash
./run.sh
```

This starts:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:3000

### Manual Start

#### Backend Only

```bash
python api/server.py
```

#### Frontend Only

```bash
cd frontend
npm run dev
```

### Command Line Interface

Run research directly from the command line:

```bash
# Basic research
python main.py "What is the current state of quantum computing?"

# With custom settings
python main.py "Analyze the EV market in 2025" --max-iterations 3 --quality-threshold 85
```

### Express Research (Faster, Less Depth)

```bash
python run_express_research.py "Your research query here"
```

## API Reference

### POST /api/research

Execute a deep research query.

**Request:**
```json
{
  "query": "Your research question here"
}
```

**Response:**
```json
{
  "report": "# Research Report\n\n...",
  "success": true,
  "summary": "Research execution summary..."
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the adoption rate of AI in US law enforcement?"}'
```

## Project Structure

```
isara-deep-research/
├── agents/                    # AI Agent implementations
│   ├── planner.py            # Research planning agent
│   ├── worker.py             # Research execution workers
│   ├── editor.py             # Report synthesis agent
│   ├── critic.py             # Quality evaluation agent
│   ├── analyst.py            # Data analysis agent
│   ├── hitl_agent.py         # Human-in-the-loop agent
│   ├── domain_experts.py     # Domain-specific expert panels
│   └── schemas.py            # Pydantic models and schemas
│
├── infrastructure/            # Tools and utilities
│   ├── perplexity_tools.py   # Perplexity search integration
│   ├── knowledge_tools.py    # LanceDB vector store
│   ├── parallel_tools.py     # Parallel API for URL extraction
│   ├── government_tools.py   # Government data sources
│   ├── daytona_tools.py      # Secure code execution
│   ├── observability.py      # LMNR/Laminar tracing
│   └── retry_utils.py        # Retry logic with backoff
│
├── api/                       # FastAPI backend
│   └── server.py             # API server
│
├── frontend/                  # Next.js frontend
│   ├── src/
│   │   ├── app/              # Next.js app router
│   │   └── components/       # React components
│   └── package.json
│
├── tests/                     # Test suite
│   ├── test_agents.py
│   ├── test_deep_research.py
│   ├── test_orchestration.py
│   └── ...
│
├── evaluation/                # Evaluation tools
│   └── slack_hitl.py         # Slack HITL integration
│
├── scripts/                   # Utility scripts
│   └── ingest_agency_list.py # Data ingestion
│
├── reports/                   # Generated research reports
├── data/                      # Data files and datasets
├── checkpoints/               # Research checkpoints
│
├── main.py                    # Main orchestration module
├── config.py                  # Configuration management
├── swarm_factory.py          # Agent factory
├── requirements.txt          # Python dependencies
├── run.sh                    # Start script
└── README.md                 # This file
```

## Configuration Options

### Model Configuration

Edit `config.py` to customize models:

```python
@dataclass
class ModelConfig:
    planner: str = "openai/claude-opus-4-5-20251101"
    worker: str = "openai/claude-opus-4-5-20251101"
    editor: str = "openai/claude-opus-4-5-20251101"
    critic: str = "openai/claude-opus-4-5-20251101"
```

### Research Depth

```python
@dataclass
class DeepResearchConfig:
    max_iterations: int = 3           # Research loop iterations
    quality_threshold: int = 80       # Min score to stop
    min_sources_per_subtask: int = 3
    target_findings_per_subtask: int = 5
    min_report_words: int = 3000
    target_report_words: int = 7000
```

### Swarm Settings

```python
@dataclass
class SwarmConfig:
    max_subtasks: int = 10    # Number of research subtasks
    max_workers: int = 5      # Parallel workers
    worker_timeout: int = 1800  # 30 minutes
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_deep_research.py

# With verbose output
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Generating PDF Reports

Convert markdown reports to PDF:

```bash
# Simple PDF generation
python generate_pdf_simple.py reports/your_report.md

# Advanced PDF with styling
python generate_pdf.py reports/your_report.md
```

## Monitoring & Observability

### Laminar (LMNR) Tracing

When `LMNR_PROJECT_API_KEY` is set, all agent operations are automatically traced:

- View traces at [laminar.ai](https://laminar.ai)
- Track latency, token usage, and errors
- Debug complex multi-agent workflows

### Logs

Logs are output to stdout with timestamps:

```bash
# Run with debug logging
DEBUG=1 python main.py "Your query"
```

## Troubleshooting

### Common Issues

**1. API Key Errors**
```
Error: PERPLEXITY_API_KEY not set
```
Solution: Ensure all required API keys are in your `.env` file.

**2. LiteLLM Connection Issues**
```
Error: Connection refused to LiteLLM proxy
```
Solution: Verify `LITELLM_API_BASE` URL is correct and accessible.

**3. Frontend Not Connecting to Backend**
```
CORS error in browser console
```
Solution: Ensure backend is running on port 8000 before starting frontend.

**4. Out of Memory**
```
MemoryError during large research
```
Solution: Reduce `max_subtasks` or `max_workers` in config.

### Getting Help

1. Check the [env.example.txt](env.example.txt) for configuration reference
2. Run `python config.py` to validate your setup
3. Check logs for detailed error messages

## Performance Tips

1. **Use LiteLLM Proxy**: Routes requests efficiently and handles rate limiting
2. **Adjust Worker Count**: More workers = faster but more API calls
3. **Set Quality Threshold**: Lower threshold = faster but less comprehensive
4. **Enable Checkpointing**: Prevents data loss on long research sessions

## License

Proprietary - All rights reserved.

## Acknowledgments

Built with:
- [Agno](https://github.com/agno-ai/agno) - Multi-agent framework
- [LiteLLM](https://github.com/BerriAI/litellm) - LLM proxy
- [Perplexity AI](https://perplexity.ai) - Search API
- [LanceDB](https://lancedb.com) - Vector database
- [Next.js](https://nextjs.org) - React framework
- [FastAPI](https://fastapi.tiangolo.com) - Python API framework
