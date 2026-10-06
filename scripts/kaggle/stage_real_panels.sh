#!/usr/bin/env bash
# Stage built real-data panels into a directory ready for a PRIVATE Kaggle dataset upload.
#
# Staging lands under runs/ (gitignored) -- the panels are third-party data and never enter git.
# Only sources whose licence permits private hosting are staged (see
# docs/research/data_candidates.csv); Freddie Mac is deliberately absent (§153: internal use).
#
# Usage:
#   scripts/kaggle/stage_real_panels.sh sba ulb_fraud
#   kaggle datasets create -p runs/kaggle_staging/fintfm-r-panels      # first time, private by default
#   kaggle datasets version -p runs/kaggle_staging/fintfm-r-panels -m "add sba"   # thereafter
set -euo pipefail
cd "$(dirname "$0")/../.."

OUT=runs/kaggle_staging/fintfm-r-panels
mkdir -p "$OUT"
for src in "$@"; do
  case "$src" in
    edgar)        from=data/edgar/panel.parquet ;;
    lendingclub)  from=data/lendingclub/panel.parquet ;;
    sba|ulb_fraud|bondora|ppdai) from=data/real/$src/panel.parquet ;;
    *) echo "refusing to stage '$src' (not in the stageable list)" >&2; exit 2 ;;
  esac
  mkdir -p "$OUT/$src"
  cp "$from" "$OUT/$src/panel.parquet"
  echo "staged $src ($(du -h "$OUT/$src/panel.parquet" | cut -f1))"
done
cat > "$OUT/dataset-metadata.json" <<'JSON'
{
  "title": "fintfm-r-panels",
  "id": "muhakabartay/fintfm-r-panels",
  "licenses": [{"name": "other"}]
}
JSON
echo "ready: $OUT (upload stays private; see docs/research/data_candidates.csv for each source's licence)"
