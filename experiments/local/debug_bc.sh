#!/bin/bash
# Launch the bc campaign under debugpy, mirroring the container the real runs use.
#
# The process blocks on --wait-for-client, so nothing executes until VS Code
# attaches with the "Attach: ConcoLLMic in Docker" configuration.
#
#   ./debug_bc.sh                     # run, parallel_num=1, out=/shared/concolic-debug
#   ./debug_bc.sh 5                   # run with parallel_num=5 (matches production)
#   ./debug_bc.sh 1 concolic-debug-2  # different output dir
#   PORT=5679 ./debug_bc.sh           # second debugger alongside the first
#
# NOTE: parallel_num defaults to 1. Above 1, solve_and_execute runs in worker
# threads and breakpoints fire on several at once, which is hard to follow.

set -euo pipefail

REPO="/mnt/NVMETank/Projects/Jef/symbolic_ai/ConcoLLMic"
RESULTS="${REPO}/experiments/benchmarks/c_c++_programs/bc/results"

PARALLEL="${1:-1}"
OUTNAME="${2:-concolic-debug}"
PORT="${PORT:-5678}"

echo "debugpy listening on ${PORT}; attach from VS Code to begin."
echo "  parallel_num=${PARALLEL}  out=/shared/${OUTNAME}"

docker run --rm -it --name "bc-debug-${PORT}" \
  --cpus 2 --memory 8g \
  -p "${PORT}:${PORT}" \
  -e ACE_MODEL="openai/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090" \
  -e LOCAL_MODEL_API_BASE="http://host.docker.internal:8000/v1" \
  -e PYTHONDONTWRITEBYTECODE=1 \
  --add-host host.docker.internal:host-gateway \
  -v "${REPO}:/concolic-agent:rw" \
  -v "${RESULTS}:/shared:rw" \
  -w /concolic-agent \
  --entrypoint bash bc-concolic:latest -c "
    pip install -q debugpy
    exec python3 -m debugpy --listen 0.0.0.0:${PORT} --wait-for-client ACE.py run \
      --project_dir /bc-instr \
      --execution /seed_execs/bc.py \
      --out /shared/${OUTNAME} \
      --timeout 10 \
      --plateau_slot 300 \
      --parallel_num ${PARALLEL}
  "
