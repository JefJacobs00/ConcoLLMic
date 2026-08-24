#!/bin/bash
# Measure throughput of the local model server.
#
#   ./measure_tps.sh bench          synthetic benchmark at ConcoLLMic's concurrency
#   ./measure_tps.sh live [seconds] actual tok/s during whatever is running now
#   ./measure_tps.sh single         one request, single-stream decode speed
#
# ConcoLLMic runs --parallel_num conversations at once (default 5) with long
# prompts and short-ish completions, so aggregate throughput at concurrency 5
# predicts wall-clock far better than a single-stream number does.

set -uo pipefail
MODE="${1:-live}"
HOST="${TPS_HOST:-127.0.0.1}"
PORT="${TPS_PORT:-8000}"
CONTAINER="${TPS_CONTAINER:-vllm_host}"
MODEL="${TPS_MODEL:-$(curl -s http://$HOST:$PORT/v1/models | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')}"

case "$MODE" in
  bench)
    # Prompt/output shape roughly matches a summarizer turn.
    docker exec "$CONTAINER" vllm bench serve \
      --model "$MODEL" --host 127.0.0.1 --port 8000 \
      --dataset-name random --random-input-len 8000 --random-output-len 500 \
      --num-prompts 20 --max-concurrency 5
    ;;

  single)
    python3 - "$HOST" "$PORT" "$MODEL" <<'PY'
import json, sys, time, urllib.request
host, port, model = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "Count from 1 to 300, one number per line."}],
    "max_tokens": 800, "temperature": 0, "stream": False,
}).encode()
req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.loads(urllib.request.urlopen(req).read())
dt = time.time() - t0
out = r["usage"]["completion_tokens"]
print(f"single-stream: {out} output tokens in {dt:.2f}s = {out/dt:.1f} tok/s")
print(f"  (prompt {r['usage']['prompt_tokens']} tokens, finish={r['choices'][0].get('finish_reason')})")
PY
    ;;

  live)
    WINDOW="${2:-30}"
    python3 - "$HOST" "$PORT" "$WINDOW" <<'PY'
import re, sys, time, urllib.request
host, port, window = sys.argv[1], sys.argv[2], float(sys.argv[3])

def scrape():
    txt = urllib.request.urlopen(f"http://{host}:{port}/metrics").read().decode()
    out = {}
    for name in ("vllm:generation_tokens_total", "vllm:prompt_tokens_total",
                 "vllm:num_requests_running", "vllm:num_requests_waiting"):
        vals = [float(m.group(1)) for m in
                re.finditer(rf"^{re.escape(name)}\{{[^}}]*}}\s+([0-9.e+-]+)$", txt, re.M)]
        if vals:
            out[name] = sum(vals)
    return out

a = scrape()
if "vllm:generation_tokens_total" not in a:
    sys.exit("no vllm:generation_tokens_total in /metrics -- check the endpoint")
time.sleep(window)
b = scrape()

gen = b["vllm:generation_tokens_total"] - a["vllm:generation_tokens_total"]
pro = b.get("vllm:prompt_tokens_total", 0) - a.get("vllm:prompt_tokens_total", 0)
print(f"over {window:.0f}s:")
print(f"  output   {gen:8.0f} tokens = {gen/window:7.1f} tok/s")
print(f"  prefill  {pro:8.0f} tokens = {pro/window:7.1f} tok/s")
print(f"  in-flight requests now: {b.get('vllm:num_requests_running', 0):.0f} running, "
      f"{b.get('vllm:num_requests_waiting', 0):.0f} queued")
if gen == 0:
    print("  (idle -- nothing generating during the window)")
PY
    ;;

  *) echo "usage: $0 {bench|live [seconds]|single}"; exit 1 ;;
esac
