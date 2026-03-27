#!/usr/bin/env bash
# --------------------------------------------------------------------------
# Run the integration stress test in a clean Docker container.
#
# This builds the paretobandit wheel from source, installs it into a fresh
# Python environment (no editable mode, no source tree), and runs the full
# integration test suite.  Catches packaging errors, missing data artifacts,
# and broken imports that in-tree unit tests would miss.
#
# Usage:
#   ./scripts/run_integration_test.sh                        # core only, Python 3.11
#   ./scripts/run_integration_test.sh 3.10                   # core only, Python 3.10
#   ./scripts/run_integration_test.sh --embeddings           # core + embeddings, Python 3.11
#   ./scripts/run_integration_test.sh --embeddings 3.10 3.11 # embeddings, multi-version
#   ./scripts/run_integration_test.sh --all 3.11             # both targets, Python 3.11
#
# Requires: Docker
# --------------------------------------------------------------------------

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_BASE="paretobandit-integration"

# Parse flags
TARGETS=()
VERSIONS=()

for arg in "$@"; do
    case "${arg}" in
        --embeddings) TARGETS+=("embeddings") ;;
        --core)       TARGETS+=("core") ;;
        --all)        TARGETS+=("core" "embeddings") ;;
        *)            VERSIONS+=("${arg}") ;;
    esac
done

# Defaults
[[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=("core")
[[ ${#VERSIONS[@]} -eq 0 ]] && VERSIONS=("3.11")

fail_count=0

for target in "${TARGETS[@]}"; do
    for pyver in "${VERSIONS[@]}"; do
        image="${IMAGE_BASE}-${target}:py${pyver}"
        echo ""
        echo "=========================================="
        echo "  [${target}] Python ${pyver}"
        echo "=========================================="
        echo ""

        if docker build \
            -f "${REPO_ROOT}/Dockerfile.integration" \
            --target "${target}" \
            --build-arg "PYTHON_VERSION=${pyver}" \
            -t "${image}" \
            "${REPO_ROOT}"; then

            if docker run --rm "${image}"; then
                echo ""
                echo "  [${target}] Python ${pyver}: PASSED"
            else
                echo ""
                echo "  [${target}] Python ${pyver}: FAILED (test run)"
                fail_count=$((fail_count + 1))
            fi
        else
            echo ""
            echo "  [${target}] Python ${pyver}: FAILED (build)"
            fail_count=$((fail_count + 1))
        fi
    done
done

echo ""
echo "=========================================="
if [ "${fail_count}" -eq 0 ]; then
    echo "  All targets passed."
else
    echo "  ${fail_count} target(s) FAILED."
    exit 1
fi
