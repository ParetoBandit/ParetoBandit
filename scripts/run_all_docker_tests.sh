#!/usr/bin/env bash
# --------------------------------------------------------------------------
# Run ALL Docker-based validation targets for the ParetoBandit project.
#
# Exercises every pip-install variant (core, embeddings, demo) across
# multiple Python versions, plus supplementary Dockerfiles that validate
# README examples, API Reference examples, and the full demo CLI.
#
# Usage:
#   ./scripts/run_all_docker_tests.sh              # default: Python 3.10 3.11 3.12
#   ./scripts/run_all_docker_tests.sh 3.11          # single Python version
#   ./scripts/run_all_docker_tests.sh 3.10 3.12     # specific versions
#
# Requires: Docker
# --------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/scripts"

VERSIONS=("$@")
[[ ${#VERSIONS[@]} -eq 0 ]] && VERSIONS=("3.10" "3.11" "3.12")

total=0
pass=0
fail=0
declare -a RESULTS=()

record_result() {
    local name="$1" status="$2"
    total=$((total + 1))
    if [[ "${status}" == "PASS" ]]; then
        pass=$((pass + 1))
        RESULTS+=("  ✓  ${name}")
    else
        fail=$((fail + 1))
        RESULTS+=("  ✗  ${name}")
    fi
}

# ------------------------------------------------------------------
# 1. Integration tests (core / embeddings / demo × Python versions)
# ------------------------------------------------------------------

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 1: Integration Tests (Dockerfile.integration)       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

if "${SCRIPT_DIR}/run_integration_test.sh" --all "${VERSIONS[@]}"; then
    for target in core embeddings demo; do
        for pyver in "${VERSIONS[@]}"; do
            record_result "integration/${target} py${pyver}" "PASS"
        done
    done
else
    # Re-run individually to identify which ones failed
    for target in core embeddings demo; do
        for pyver in "${VERSIONS[@]}"; do
            if "${SCRIPT_DIR}/run_integration_test.sh" "--${target}" "${pyver}" 2>/dev/null; then
                record_result "integration/${target} py${pyver}" "PASS"
            else
                record_result "integration/${target} py${pyver}" "FAIL"
            fi
        done
    done
fi

# ------------------------------------------------------------------
# 2. README examples (Dockerfile.readme)
# ------------------------------------------------------------------

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 2: README Examples (Dockerfile.readme)              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if docker build -f "${REPO_ROOT}/Dockerfile.readme" \
    -t paretobandit-readme "${REPO_ROOT}" \
    && docker run --rm paretobandit-readme; then
    record_result "Dockerfile.readme" "PASS"
else
    record_result "Dockerfile.readme" "FAIL"
fi

# ------------------------------------------------------------------
# 3. API Reference examples (Dockerfile.examples)
# ------------------------------------------------------------------

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 3: API Reference Examples (Dockerfile.examples)     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if docker build -f "${REPO_ROOT}/Dockerfile.examples" \
    -t paretobandit-examples "${REPO_ROOT}" \
    && docker run --rm paretobandit-examples; then
    record_result "Dockerfile.examples" "PASS"
else
    record_result "Dockerfile.examples" "FAIL"
fi

# ------------------------------------------------------------------
# 4. Full demo CLI (Dockerfile.demo)
# ------------------------------------------------------------------

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Phase 4: Demo CLI (Dockerfile.demo)                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

DEMO_OUTPUT="${REPO_ROOT}/docker_results"
mkdir -p "${DEMO_OUTPUT}"

if docker build -f "${REPO_ROOT}/Dockerfile.demo" \
    -t paretobandit-demo "${REPO_ROOT}" \
    && docker run --rm -v "${DEMO_OUTPUT}:/output" paretobandit-demo; then
    record_result "Dockerfile.demo" "PASS"
else
    record_result "Dockerfile.demo" "FAIL"
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SUMMARY                                                   ║"
echo "╠══════════════════════════════════════════════════════════════╣"
for r in "${RESULTS[@]}"; do
    printf "║  %-56s  ║\n" "${r}"
done
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Total: %d    Passed: %d    Failed: %d%-20s║\n" \
    "${total}" "${pass}" "${fail}" ""
echo "╚══════════════════════════════════════════════════════════════╝"

if [[ "${fail}" -gt 0 ]]; then
    exit 1
fi
