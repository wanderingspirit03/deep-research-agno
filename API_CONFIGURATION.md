# LiteLLM API Configuration Guide

## Summary of Changes Made

### 1. PDF Generation Fixed
- **Issue**: Unicode character encoding errors (bullet points, special quotes)
- **Solution**: Added DejaVu font support with comprehensive character replacement
- **Files Modified**: `generate_pdf_simple.py`, `generate_pdf.py`

### 2. Editor System Prompts Improved
- **Goal**: Better prose quality, not just comprehensive reports
- **Changes**: Added writing craft guidelines, narrative focus, quality checklist
- **Files Modified**: `agents/editor.py`

### 3. Fallback Report Generation Added
- **Issue**: When Editor agent times out (504), no report was generated
- **Solution**: Added `_generate_simple_fallback_report()` method
- **Files Modified**: `main.py`

### 4. Mock Mode Added
- **Purpose**: Test system without API calls
- **Usage**: `python3 main.py "query" --mock --output report.md`
- **Files Modified**: `main.py`

---

## API Configuration

### Required Environment Variables

```bash
# In .env file
LITELLM_API_BASE=https://model-proxy.development.research-platform.isara.io/v1
LITELLM_API_KEY=sk-xxxxxxxxxxxxxx
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxx
```

### ⚠️ Critical: The `/v1` Suffix

The API base URL **MUST** include `/v1` at the end:

```bash
# ✅ CORRECT
LITELLM_API_BASE=https://model-proxy.development.research-platform.isara.io/v1

# ❌ WRONG (will return 403 Forbidden)
LITELLM_API_BASE=https://model-proxy.development.research-platform.isara.io
```

---

## How to Get a Valid API Key

### Option 1: Using Cave (Recommended)

The API key must be minted through the `cave` library with proper AWS authentication:

```python
from cave import mint_llm_key, get_session, get_username
from cave.constants import LITELLM_PROXY_URL, MINT_KEY_LAMBDA_URL

# First: isara login aws
session = get_session()
username = get_username(session)

auth = mint_llm_key(
    experiment_id='deep-research-swarm',
    username=username,
    lambda_url=MINT_KEY_LAMBDA_URL,
    proxy_url=LITELLM_PROXY_URL,
    session=session,
)

print(f'API Key: {auth.api_key}')
print(f'Base URL: {auth.base_url}')
```

### Option 2: Manual Key with Model Access

If creating keys manually in LiteLLM admin panel:
1. Create a new virtual key
2. **Assign models** to the key (critical - empty models = 403)
3. Add models like: `claude-haiku-4-5-20251001`, `gpt-5-mini-2025-08-07`

---

## Available Models

| Model ID | Description |
|----------|-------------|
| `claude-haiku-4-5-20251001` | Claude Haiku (fast) |
| `claude-sonnet-4-5-20250929` | Claude Sonnet |
| `claude-opus-4-1-20250805` | Claude Opus |
| `claude-opus-4-5-20251101` | Claude Opus (newer) |
| `gpt-5-mini-2025-08-07` | GPT-5 Mini |
| `gpt-5-nano-2025-08-07` | GPT-5 Nano |
| `gpt-5-2025-08-07` | GPT-5 |
| `gpt-5-pro-2025-10-06` | GPT-5 Pro |
| `gpt-5-codex` | GPT-5 Codex |
| `o3-deep-research-2025-06-26` | O3 Deep Research |

---

## Testing the API

### Quick Test with curl

```bash
curl -X POST "https://model-proxy.development.research-platform.isara.io/v1/chat/completions" \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-haiku-4-5-20251001",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 30
  }'
```

### Test with Python

```python
from dotenv import load_dotenv
import os
import litellm

load_dotenv()
litellm.drop_params = True

response = litellm.completion(
    model='claude-haiku-4-5-20251001',
    messages=[{'role': 'user', 'content': 'Hello'}],
    api_base=os.getenv('LITELLM_API_BASE'),
    api_key=os.getenv('LITELLM_API_KEY'),
    max_tokens=30
)
print(response.choices[0].message.content)
```

### Check Key Info

```bash
curl -s "https://model-proxy.development.research-platform.isara.io/key/info" \
  -H "Authorization: Bearer $LITELLM_API_KEY" | python3 -m json.tool
```

---

## Troubleshooting

### 403 Forbidden Error

**Possible Causes:**
1. Missing `/v1` in base URL
2. API key has no models assigned (`"models": []`)
3. API key expired or invalid

**Debug Steps:**
```bash
# Check key info
curl -s "https://model-proxy.development.research-platform.isara.io/key/info" \
  -H "Authorization: Bearer YOUR_KEY"

# Look for:
# - "models": [] means NO ACCESS
# - "expires": check if key is still valid
```

### 504 Gateway Timeout

**Cause:** Long-running synthesis requests timeout at the proxy level (usually 60s)

**Solution:** The system now has fallback report generation. If Editor times out, a report is still generated from collected findings.

---

## Running Deep Research

### Standard Mode
```bash
python3 main.py "Your research query" --output report.md
```

### Simple/Fast Mode
```bash
python3 main.py "Your research query" --simple --output report.md
```

### Mock Mode (No API)
```bash
python3 main.py "Your research query" --mock --output report.md
```

---

## File Structure

```
.env                    # API keys (LITELLM_API_BASE, LITELLM_API_KEY)
config.py               # Configuration settings
main.py                 # Main entry point
agents/
  planner.py           # Research planning agent
  worker.py            # Research execution agent  
  editor.py            # Report synthesis agent
  critic.py            # Quality review agent
  domain_experts.py    # Expert analysis agents
infrastructure/
  perplexity_tools.py  # Search tools
  knowledge_tools.py   # Knowledge base tools
  parallel_tools.py    # URL extraction tools
```

---

*Last Updated: December 5, 2025*

