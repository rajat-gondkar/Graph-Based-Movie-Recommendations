#!/usr/bin/env bash
# Overnight run for Phase 2: extract themes for every movie that is not cached yet, make a second pass
# for any leftovers (API hiccups), then normalize. Safe to stop and re-run: finished movies are never paid for twice.
#
# Usage, from the movie-kg-recsys folder (keeps the Mac awake until it finishes):
#     caffeinate -i bash scripts/overnight_extract.sh
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate || exit 1
stamp=$(date +%Y%m%d_%H%M)
log="results/extraction_run_${stamp}.log"

python scripts/03_extract_themes.py --yes --retry-failed 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "Extraction stopped with an error (exit code $status). Normalization was skipped." | tee -a "$log"
    exit "$status"
fi

echo "--- second pass (picks up anything that hit an API error) ---" | tee -a "$log"
python scripts/03_extract_themes.py --yes --retry-failed 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
if [ "$status" -ne 0 ]; then
    echo "Second pass stopped with an error (exit code $status). Normalization was skipped." | tee -a "$log"
    exit "$status"
fi

python scripts/04_normalize_themes.py 2>&1 | tee "results/normalize_run_${stamp}.log"
echo "Finished at $(date)." | tee -a "$log"
