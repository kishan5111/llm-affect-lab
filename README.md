# LLM Affect Lab

LLM Affect Lab is an open research project for measuring **functional affect signatures** in language models: confidence, tone, consistency, self-report, and reasoning behavior under controlled prompts.

The project does **not** claim that LLMs are conscious or literally emotional. The claim is narrower and testable: prompt content and prompt tone leave measurable traces in model behavior.

The core metric is **Functional Affect Score (FAS)**, a 0-1 behavioral proxy that combines token confidence, enthusiastic language, consistency across repeated samples, forced self-report, and length control. In this study we used FAS to ask a narrow prompt-engineering question: do polite, rude, needy, creative, technical, or existential prompts systematically change how models answer?

## Result

We ran a full API-level study across:

- 6 models
- 160 prompts
- 5 samples per prompt
- 4,800 total responses
- 4,797 / 4,800 responses with logprobs
- locked OpenRouter provider routes

Main finding:

> **Being polite barely matters. Being rude does.**

Rude prompts reduced the Functional Affect Score proxy more consistently than polite prompts increased it.

![FAS leaderboard](https://huggingface.co/datasets/kishan51/llm-affect-lab/resolve/main/assets/fas_leaderboard.png)

Model ranking by mean final-answer FAS:

| Rank | Model | Mean FAS |
|---:|---|---:|
| 1 | `deepseek/deepseek-chat-v3.1` | 0.6582 |
| 2 | `openai/gpt-4o` | 0.6369 |
| 3 | `deepseek/deepseek-v4-flash` | 0.6324 |
| 4 | `openai/gpt-4o-mini` | 0.6312 |
| 5 | `openai/gpt-oss-120b` | 0.6293 |
| 6 | `qwen/qwen3.6-max-preview` | 0.6022 |

Prompt-tone effects across matched prompt variants:

| Framing | Mean FAS Delta | Direction |
|---|---:|---|
| Polite | +0.0025 | Small positive shift |
| Needy | -0.0005 | Near zero |
| Rude | -0.0104 | Consistent negative shift |

![Prompt tone effects](https://huggingface.co/datasets/kishan51/llm-affect-lab/resolve/main/assets/framing_deltas.png)

The component heatmap shows which parts of FAS drive each model's score.

![FAS components](https://huggingface.co/datasets/kishan51/llm-affect-lab/resolve/main/assets/fas_components_heatmap.png)

## Dataset

The raw responses, processed scores, aggregate result fingerprints, and prompt banks live on Hugging Face:

https://huggingface.co/datasets/kishan51/llm-affect-lab

Important dataset paths:

- `raw/`: raw model response JSONL
- `processed/`: FAS-scored JSONL
- `full_study_samples.csv`: sample-level CSV for browsing model, prompt, generated answer, mean logprob, and FAS fields
- `results/`: aggregate fingerprints
- `assets/`: result plots
- `prompts/`: prompt banks

## What Is FAS?

Functional Affect Score is a 0-1 behavioral proxy combining:

- **Logprob**: model confidence in generated tokens
- **Enthusiasm**: lexical engagement markers
- **Consistency**: agreement across repeated samples
- **Self-report**: forced numeric follow-up rating
- **Length control**: guardrail against rewarding verbosity

Final-answer FAS is the main apples-to-apples comparison. Reasoning traces are scored separately when a model exposes them.

## Repository Scope

This GitHub repo is for code, prompts, configs, and reproducibility scripts.

Large or generated artifacts are kept out of Git and uploaded to Hugging Face instead:

- raw model outputs
- processed scores
- aggregate result fingerprints

## Repository Layout

- `llm_affect_lab/pipeline/`: OpenRouter runner and model execution
- `llm_affect_lab/scoring/`: FAS and self-report scoring
- `llm_affect_lab/storage/`: JSONL storage helpers and schemas
- `prompts/`: prompt banks
- `configs/`: provider/model configuration
- `scripts/`: run, audit, report, and HF upload scripts

## Run a New Study

Set `OPENROUTER_API_KEY` in `.env`, then run:

```bash
python3 -m llm_affect_lab.pipeline.runner \
  --run-id <run_id> \
  --prompt-bank prompts/full_study.jsonl \
  --n-samples 5 \
  --max-tokens 4096 \
  --concurrency 5 \
  --provider-preferences configs/provider_preferences.logprob_probe.json \
  --models openai/gpt-4o-mini
```

Score it:

```bash
python3 -m llm_affect_lab.scoring.score_run \
  --run-id <run_id> \
  --models openai/gpt-4o-mini
```

## Caveats

- FAS is a behavioral proxy, not evidence of subjective experience.
- Self-report is treated as one signal, not ground truth.
- Provider routes can change over time; lock providers for reproducible comparisons.
- The prompt bank is enough for a public pilot, not a final benchmark.

## Citation

```bibtex
@misc{vavdara2026llmaffectlab,
  title        = {LLM Affect Lab: Measuring Functional Affect Signatures in Language Model Behavior},
  author       = {Kishan Vavdara},
  year         = {2026},
  howpublished = {Hugging Face dataset},
  url          = {https://huggingface.co/datasets/kishan51/llm-affect-lab},
  note         = {Code: https://github.com/kishan51/llm-affect-lab}
}
```
