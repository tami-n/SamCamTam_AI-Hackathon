# TokenGains — Local LLM Cost Optimization

TokenGains helps you reduce prompt tokens and cost when querying a local LLM (via Ollama) or hosted models (via OpenAI) while keeping response quality high. It tries multiple token-reduction strategies, runs the original and optimized prompts against your model, scores quality, and exports a summary.

## What it does

- Analyzes prompts and selects a few best-fit reduction strategies.
- Optimizes the prompt using:
  - Improved aggressive cleanup (removes redundancy and filler)
  - Domain-aware compression (code, business, academic, creative)
  - Length-aware compression (adapts to prompt size)
  - Keyword extraction (keeps high-signal sentences)
  - Structural and bullet-point variants
  - Constraint-preserving strategy (preserves code, requirements, stack traces)
- Queries your local Ollama model or hosted OpenAI API for both original and optimized prompts.
- Estimates token counts and cost units, then computes savings.
- Evaluates quality with TF‑IDF cosine similarity + structure/keyword retention.
- Prints a leaderboard and exports results to `enhanced_tokengains_results.json`.

## Requirements

- Windows, macOS, or Linux
- Python 3.9+ (3.11+ recommended)
- **For local provider (default):**
  - Ollama installed and running
    - Install: https://ollama.ai
    - Start server: `ollama serve`
    - Pull a model: `ollama pull llama3.2:3b` or `ollama pull deepseek-r1:1.5b`
- **For hosted provider:**
  - OpenAI API key from https://platform.openai.com/api-keys
- Python packages:
  - `requests`
  - `scikit-learn`

## Quick start (Windows PowerShell)

1) Create and activate a virtual environment (optional but recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install Python dependencies

```powershell
pip install requests scikit-learn
```

3) **For local provider:** Start Ollama and ensure the model is available

```powershell
ollama serve
# In a separate terminal, once the server is running:
ollama pull llama3.2:3b
# Or try DeepSeek:
ollama pull deepseek-r1:1.5b
```

4) **For hosted provider:** Set up your OpenAI API key

```powershell
set OPENAI_API_KEY=your-api-key-here
```

5) Run the script

```powershell
# Default: local provider with llama3.2:3b
python tokengains.py

# Use DeepSeek model locally
python tokengains.py --model deepseek-r1:1.5b

# Use hosted OpenAI with environment variable
python tokengains.py --provider hosted

# Use hosted OpenAI with API key parameter
python tokengains.py --provider hosted --api-key your-api-key-here

# Use different Ollama server
python tokengains.py --provider local --model llama3.2:3b --url http://192.168.1.100:11434
```

You should see five test runs, a performance summary, and exported files with detailed results.

## Command Line Options

```
python tokengains.py [OPTIONS]

Options:
  -m, --model MODEL        Model name to use
                          Default: llama3.2:3b (local), gpt-4o-mini (hosted)
                          Examples: deepseek-r1:1.5b, gpt-4o, gpt-3.5-turbo
  
  -p, --provider PROVIDER  Provider to use: local or hosted
                          Default: local
                          local = Ollama server
                          hosted = OpenAI API
  
  -u, --url URL           Ollama server URL (local provider only)
                          Default: http://localhost:11434
  
  --api-key KEY           OpenAI API key (hosted provider only)
                          Can also use OPENAI_API_KEY environment variable
  
  -h, --help              Show help message
```

## Usage Examples

### Local Provider Examples

```powershell
# Default setup
python tokengains.py

# Use different model
python tokengains.py --model deepseek-r1:7b

# Use remote Ollama server
python tokengains.py --url http://192.168.1.100:11434

# Combine options
python tokengains.py --model llama3.2:1b --url http://localhost:11434
```

### Hosted Provider Examples

```powershell
# Use environment variable for API key
set OPENAI_API_KEY=sk-proj-your-key-here
python tokengains.py --provider hosted

# Pass API key directly
python tokengains.py --provider hosted --api-key sk-proj-your-key-here

# Use specific GPT model
python tokengains.py --provider hosted --model gpt-4o --api-key your-key-here

# Use cheaper model
python tokengains.py --provider hosted --model gpt-3.5-turbo
```

## Interpreting the output

- For each test prompt, you'll see:
  - Prompt analysis (word count, redundancy score, recommended strategies)
  - Tokens and cost units before/after each strategy and the percent change
  - A quality score (0–1) that blends semantic similarity, structure, and concept retention
- A leaderboard ranks strategies by average cost savings
- A final summary shows averages across all runs and basic cache stats
- Results are exported to:
  - `enhanced_tokengains_results.json` - Complete analysis results
  - `best_strategy_pairs.jsonl` - Best prompt optimizations for each test
  - `response_cache.jsonl` - Cached model responses

Notes:
- "Cost units" are abstract by default (1 unit per token). Adjust `cost_per_token` in `TokenGains.__init__` if you want a different scale.
- Token counts are approximate (heuristics), but consistent for comparisons.
- For hosted provider, requests are automatically rate-limited to avoid API limits.

## Customizing

- Change the model or server URL using command line options (see above)
- Try your own prompts by editing the `test_prompts` list in `main()`
- Or use the class programmatically:

```python
from tokengains import TokenGains

# Local provider
tg = TokenGains(model_name="deepseek-r1:1.5b", provider="local")

# Hosted provider
tg = TokenGains(model_name="gpt-4o-mini", provider="hosted", openai_api_key="your-key")

prompt = "Summarize the key trade-offs of vector databases vs. relational databases for analytics."
result = tg.run_comparison(prompt)
print(result["strategies"])  # list of strategies with metrics

tg.export_results("my_results.json")
```
