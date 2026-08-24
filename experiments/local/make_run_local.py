#!/usr/bin/env python3
"""
Generate a local-model variant of a benchmark's run.sh.

The only change is to `setup_concolic_environment`: upstream resets
/concolic-agent to origin/main, which would discard the bind-mounted working
tree carrying local-model support. Everything else -- the ACE.py invocation,
the plateau termination, the coverage replay -- is left byte-identical so
results stay comparable to the published run.

Usage: ./make_run_local.py <benchmark> [<benchmark> ...]
"""
import pathlib
import sys

BENCH_DIR = pathlib.Path(__file__).parent / ".." / "benchmarks" / "c_c++_programs"
OUT_DIR = pathlib.Path(__file__).parent

OLD = """    git fetch origin
    git reset --hard origin/main    # pull the latest version
"""
NEW = """    # NOTE: upstream run.sh does `git fetch && git reset --hard origin/main` here.
    # We deliberately do not: /concolic-agent is the bind-mounted working tree
    # that carries the local-model support this run depends on.
"""
LOG_OLD = """    echo "git version: $git_version" >> ${SHARED_DIR}/execution_command.log
"""
LOG_NEW = """    echo "git version: $git_version (local working tree, not reset)" >> ${SHARED_DIR}/execution_command.log
    echo "model: ${ACE_MODEL} via ${LOCAL_MODEL_API_BASE}" >> ${SHARED_DIR}/execution_command.log
"""


def generate(benchmark: str) -> pathlib.Path:
    src = (BENCH_DIR / benchmark / "run.sh").resolve()
    if not src.is_file():
        sys.exit(f"no run.sh for benchmark '{benchmark}' at {src}")

    s = src.read_text()
    if OLD not in s:
        sys.exit(f"{src}: git reset block not found verbatim -- upstream changed shape")
    s = s.replace(OLD, NEW).replace(LOG_OLD, LOG_NEW)

    dst = OUT_DIR / f"run_local.{benchmark}.sh"
    dst.write_text(s)
    dst.chmod(0o755)
    return dst


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for b in sys.argv[1:]:
        print(f"wrote {generate(b)}")
