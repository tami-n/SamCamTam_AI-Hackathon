# TokenGains — Local LLM Cost Optimization

TokenGains helps you reduce prompt tokens and cost when querying a local LLM (via Ollama) while keeping response quality high. It tries multiple token-reduction strategies, runs the original and optimized prompts against your model, scores quality, and exports a summary.

## What it does

- Analyzes prompts and selects a few best-fit reduction strategies.
- Optimizes the prompt using:
  - Improved aggressive cleanup (removes redundancy and filler)
  - Domain-aware compression (code, business, academic, creative)
  - Length-aware compression (adapts to prompt size)
  - Keyword extraction (keeps high-signal sentences)
  - Structural and bullet-point variants
- Queries your local Ollama model for both original and optimized prompts.
- Estimates token counts and cost units, then computes savings.
- Evaluates quality with TF‑IDF cosine similarity + structure/keyword retention.
- Prints a leaderboard and exports results to `enhanced_tokengains_results.json`.

## Requirements

- Windows, macOS, or Linux
- Python 3.9+ (3.11+ recommended)
- Ollama installed and running
  - Install: https://ollama.ai
  - Start server: `ollama serve`
  - Pull the default model used by this repo: `ollama pull llama3.2:3b`
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

3) Start Ollama and ensure the model is available

```powershell
ollama serve
# In a separate terminal, once the server is running:
ollama pull llama3.2:3b
```

4) Run the script

```powershell
python .\tokengains.py
```

You should see five test runs, a performance summary, and an exported file `enhanced_tokengains_results.json` with details.

## Interpreting the output

- For each test prompt, you’ll see:
  - Tokens and cost units before/after each strategy and the percent change
  - A quality score (0–1) that blends semantic similarity, structure, and concept retention
- A leaderboard ranks strategies by average cost savings
- A final summary shows averages across all runs and basic cache stats
- Results are exported to `enhanced_tokengains_results.json`

Notes:
- “Cost units” are abstract by default (1 unit per token). Adjust `cost_per_token` in `TokenGains.__init__` if you want a different scale.
- Token counts are approximate (heuristics), but consistent for comparisons.

## Customizing

- Change the model or server URL:
  - In `tokengains.py`, update `TokenGains(model_name="llama3.2:3b", ollama_url="http://localhost:11434")`
- Try your own prompts:
  - Edit the `test_prompts` list in `main()`
  - Or use the class programmatically:

```python
from tokengains import TokenGains

tg = TokenGains(model_name="llama3.2:3b", ollama_url="http://localhost:11434")
prompt = "Summarize the key trade-offs of vector databases vs. relational databases for analytics."
result = tg.run_comparison(prompt)
print(result["strategies"])  # list of strategies with metrics

tg.export_results("my_results.json")
```

- Tune aggressiveness/quality:
  - Edit or add strategies by extending `TokenReductionStrategy`
  - Adjust `select_best_strategies()` for selection rules
  - Adjust weightings in `evaluate_response_quality()` to bias toward quality or savings

## Troubleshooting

- Cannot connect to Ollama:
  - Ensure `ollama serve` is running and `http://localhost:11434` is reachable
  - Pull the model first: `ollama pull llama3.2:3b`
  - If you changed the model name in code, make sure it’s pulled and spelled correctly
- `scikit-learn` install issues:
  - Upgrade pip: `python -m pip install --upgrade pip`
  - Try again in a fresh virtual environment
- Empty or low quality scores:
  - Quality uses TF‑IDF similarity; very short or highly rephrased responses can score lower
  - Adjust weights in `evaluate_response_quality()` if needed

## Files

- `tokengains.py` — main script with strategies, scoring, and execution
- `enhanced_tokengains_results.json` — latest exported results (created on run)
- `tokengains_results.json` — legacy or alternative results (if present)
- `input.txt` — optional scratchpad (not used by the script)

## Roadmap ideas

- Add CLI flags to pass a prompt/file and set model/URL without editing code
- Support multiple models and batch runs via config
- More robust tokenization (e.g., model-specific tokenizers)
- Optional OpenAI/compatible API support
- Persist and visualize runs over time
