#!/bin/bash -l
# Runs one ConcoLLMic benchmark against a locally-served model, all on one node.
#
#   sbatch run_bc.slurm
#   BENCHMARK=confetti TIMEOUT=2h sbatch run_bc.slurm
#
# Slurm kills the job's process tree at the end, so vLLM and the podman service
# started here need no explicit cleanup.

#SBATCH --job-name=concollmic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out
### FILL IN the partition, then drop the extra hashes:
#SBATCH --partition=gpu_b200
#SBATCH --account=lp_distrinet_gent


set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
[[ -f docker-compose.local.yml ]] || { echo "submit from experiments/local/" >&2; exit 1; }

BENCHMARK="${BENCHMARK:-bc}"                     # bc | confetti
SERVED_NAME="${SERVED_NAME:-Qwen/Qwen3.8-27B}"   # must match /v1/models exactly
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"          # keep equal to --gpus-per-node
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"          # smaller window => more KV cache
TIMEOUT="${TIMEOUT:-24h}"                         # upper bound; plateau ends it sooner
VENV="${VENV:-/data/leuven/367/vsc36781/ai/ConcoLLMic/.venv/}"            # the venv holding vllm

#export HF_HOME="${HF_HOME:-${VSC_SCRATCH:-$HOME}/hf_cache}"   # ~54 GB of weights
export ACE_MODEL="openai/${SERVED_NAME}"
export LOCAL_MODEL_API_BASE="http://localhost:8000/v1"
export TIMEOUT
export CONDA_PREFIX="/data/leuven/367/vsc36781/miniconda3"
export CUDA_HOME="$CONDA_PREFIX"
export HF_HOME="/scratch/leuven/367/vsc36781/hf_models"

export FLASHINFER_EXTRA_LDFLAGS="\
-L$CONDA_PREFIX/targets/x86_64-linux/lib \
-L$CONDA_PREFIX/targets/x86_64-linux/lib/stubs"

export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export PATH="/data/leuven/367/vsc36781/ai/ConcoLLMic/.venv/bin:$PATH"


# Compose talks to podman over a socket that no systemd user session will start.
# Compute nodes have no XDG_RUNTIME_DIR either, so podman's default socket path
# lands in a directory nothing has created. Make one that dies with the job.
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/podman-$(id -u)-${SLURM_JOB_ID}}"
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"

PODMAN_SOCK="$(podman info --format '{{.Host.RemoteSocket.Path}}')"
mkdir -p "$(dirname "$PODMAN_SOCK")"
export DOCKER_HOST="unix://$PODMAN_SOCK"
podman system service --time=0 "$DOCKER_HOST" &

for _ in {1..30}; do [[ -S "$PODMAN_SOCK" ]] && break; sleep 1; done
[[ -S "$PODMAN_SOCK" ]] || { echo "podman socket never appeared: $PODMAN_SOCK" >&2; exit 1; }

podman compose -f docker-compose.local.yml build concolic "${BENCHMARK}-concolic"

# vLLM runs natively, so it sees the GPUs Slurm allocated directly. Calling the
# venv binary avoids uv having to resolve a project from inside the repo.
"$VENV/bin/vllm" serve "$SERVED_NAME" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    > "vllm-${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

# Without this check a dead vLLM leaves the job idling on the GPU until walltime.
echo "waiting for vLLM (log: vllm-${SLURM_JOB_ID}.log) ..."
until curl -sf http://localhost:8000/v1/models | grep -q "$SERVED_NAME"; do
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vLLM died, see vllm-${SLURM_JOB_ID}.log" >&2; exit 1; }
    sleep 15
done
echo "vLLM ready, starting $BENCHMARK"

# --no-deps: the `concolic` service only exists to build a base image.
podman compose -f docker-compose.local.yml up --no-deps --force-recreate "${BENCHMARK}-concolic"

echo "results in ../benchmarks/c_c++_programs/${BENCHMARK}/results/"
