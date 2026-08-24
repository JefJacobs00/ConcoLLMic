# Running ConcoLLMic benchmarks against a local model

Reproduces a paper benchmark run, pointed at a local OpenAI-compatible server
instead of the Anthropic API. Resource limits, seed harness, `--plateau_slot 30`
termination and coverage replay are unchanged from the artifact's `run.sh`.

Currently wired up: `bc-concolic`, `confetti-concolic`. Add another with
`./make_run_local.py <benchmark>` plus a service block copied from an existing one.

## 1. Serve the model

The agents call with `tool_choice="required"` on every turn, so the server must
honour forced tool calls. vLLM does this via guided decoding; **ollama does not**
(it silently returns prose instead), costing the agent a re-prompt each time.

```bash
docker run --rm -p 8000:8000 --gpus all --name vllm_host \
  --volume /path/to/hf_cache:/root/hf_cache \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3.5-9B \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

- **`--tool-call-parser` must match the model's chat template.** Qwen3.5 emits
  XML-style calls (`<tool_call><function=…><parameter=…>`) → `qwen3_xml`
  (`qwen3_coder` is an alias for the same class). Hermes-style JSON models use
  `hermes`. Check the template rather than guessing:
  `grep -o '<tool_call>' $(model)/tokenizer_config.json`.
- Without it, every agent call fails with
  `tool_choice="required" requires --tool-call-parser to be set`.
- **`--reasoning-parser`**: needed for thinking models. Qwen3.5's template carries
  `<think>` / `enable_thinking` / `reasoning_content`; without the parser that
  reasoning text lands in `content` and is appended to the agent's message thread,
  inflating every later prompt in the conversation. vLLM 0.27 has no
  `--chat-template-kwargs` serve flag, so thinking cannot be disabled server-side.
- **Output budget**: reasoning eats into `max_tokens`. If responses come back
  truncated before the tool call, raise `LOCAL_MODEL_MAX_OUTPUT_TOKENS`
  (default 16384) on the compose service.
- **Context**: vLLM already serves the model's native maximum. Consider
  *lowering* it (`--max-model-len 65536`) rather than raising it — ConcoLLMic runs
  5 conversations concurrently and a smaller window fits more of them in KV cache.
  The summarizer bails at 188k tokens anyway
  ([agent_summarizer.py:579](../../app/agents/agent_summarizer.py#L579)).
- A prompt exceeding the served context returns `BadRequestError`, which tenacity
  does **not** retry — that round dies.

## 2. Run

```bash
cd experiments/local

ACE_MODEL=openai/Qwen/Qwen3.5-9B TIMEOUT=48h \
  docker compose -f docker-compose.local.yml up bc-concolic
```

`ACE_MODEL` names a model registered in [register.py](../../app/model/register.py); its `served_name` must match the id from `curl localhost:8000/v1/models`. Add new local models as classes in [local.py](../../app/model/local.py).
`TIMEOUT` is only an upper bound; the run self-terminates after 30 minutes without
new coverage. Results land in
`../benchmarks/c_c++_programs/<benchmark>/results/concolic-<timestamp>-<host>/`:

- `coverage_summary.csv` — time series of line/branch coverage
- `output/queue/id:*.yaml` — generated test cases
- `run_data.log` — token usage and (zero) cost
- `replay.log`, `output/gcov/` — per-line coverage

## 3. What the paper got (Table 5, claude-3.7-sonnet)

| | bc | confetti |
|---|---|---|
| Instrumented LoC | 1,888 (max file 2,703) | 678 (max file 2,035) |
| Test inputs generated | 281 avg | 119 avg |
| Runtime to plateau | 4.56 h avg | 2.12 h avg |
| Total cost | $48.50 | $20.23 |

Branch coverage itself is only plotted (Figure 5), not tabulated — read the target
off that figure, or run the bundled KLEE/AFL++ services for a local baseline.

Interpretation caveats:

- **bc** is the paper's own worked example and a subject where ConcoLLMic beat every
  DSE tool, so it's the honest comparison. It is also the *longer* of the two.
- **confetti** is faster but is the one subject where ConcoLLMic did **not** beat
  every DSE tool, and it is marked `*` (released after claude-3.7's cutoff) — a
  model trained later has likely seen it, which flatters a local run.
- Paper runtimes are claude-3.7-sonnet latencies. Local wall-clock will differ in
  both directions: faster per call, more calls if the model fumbles tool use.

## Differences from the upstream run

- `run_local.<benchmark>.sh` drops upstream's `git fetch && git reset --hard
  origin/main`, which would wipe the bind-mounted working tree this setup needs.
- The repo root is mounted over `/concolic-agent` instead of the image's clone.
- The container reaches the host via `host.docker.internal`.
