#!/usr/bin/env bash
#
# Sets up branch protection rules on the main branch.
# Run this AFTER transferring the repo to the ParetoBandit org
# and making it public.
#
# Usage:  bash .github/setup_branch_protection.sh
#
# Requires: gh CLI authenticated with admin access to the repo.

set -euo pipefail

REPO="ParetoBandit/banditGPT"

echo "Setting branch protection on ${REPO}:main ..."

gh api "repos/${REPO}/branches/main/protection" \
  --method PUT \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": [
      "test (3.10)",
      "test (3.11)",
      "test (3.12)",
      "lint-typecheck",
      "build"
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
EOF

echo "Branch protection configured successfully."
echo ""
echo "Rules applied:"
echo "  - PRs required to merge into main"
echo "  - 1 approving review required (stale reviews dismissed)"
echo "  - CI status checks must pass (test matrix + lint + build)"
echo "  - Strict mode: branch must be up-to-date with main before merging"
echo "  - Force pushes and branch deletion disabled"
