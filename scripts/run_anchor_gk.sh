#!/usr/bin/env bash
# File: ./scripts/AnchorGK/run_anchor_gk.sh
# Purpose: Grid-run AnchorGK pipeline with many hyperparameters
# Usage examples:
#   ./scripts/AnchorGK/run_anchor_gk.sh UScoast
#   ./scripts/AnchorGK/run_anchor_gk.sh Tehiku
#   ./scripts/AnchorGK/run_anchor_gk.sh Shenzhen
# Notes:
#   - Requires: main.py + AnchorGK_final_clean.py in project root
#   - You can pre-set CUDA via: export CUDA_VISIBLE_DEVICES=0

set -euo pipefail

# Python interpreter (override with: PY=/path/to/python ./run_anchor_gk.sh ...)
PY="${PY:-python}"

# Dataset argument: UScoast|NDBC|Tehiku|Shenzhen (default: UScoast)
DATASET="${1:-UScoast}"

# -------------------------------
# Hyper-parameter grids
# -------------------------------
K_LIST=(3 5 7)
NUM_SUBDIV_LIST=(3 5)
KING_SELECT_LIST=(3 5 7)
WEIGHT_LIST=(0.10 0.20 0.30)
SEED_LIST=(42 123 2025)

# -------------------------------
# Paths per dataset
# -------------------------------
BASE_OUT="outputs"

case "${DATASET}" in
  UScoast|NDBC)
    NDBC_VALUES="data/NDBC/all.npy"
    NDBC_STATIONS="data/NDBC/Station_info_edit.csv"
    ROOT_OUT="${BASE_OUT}/UScoast_grid"
    DS_ARG=(--dataset UScoast --ndbc-values "${NDBC_VALUES}" --ndbc-stations "${NDBC_STATIONS}")
    ;;

  Tehiku)
    TEHIKU_COORD="data/Tehiku/COORD.xlsx"
    TEHIKU_VALUE_DIR="data/Tehiku/value"
    ROOT_OUT="${BASE_OUT}/Tehiku_grid"
    DS_ARG=(--dataset Tehiku --tehiku-coord-xlsx "${TEHIKU_COORD}" --tehiku-value-dir "${TEHIKU_VALUE_DIR}")
    ;;

  Shenzhen)
    SHENZHEN_VALUE_DIR="data/Shenzhen/Value"
    SHENZHEN_STATION_CSV="data/Shenzhen/Station.csv"
    ROOT_OUT="${BASE_OUT}/Shenzhen_grid"
    DS_ARG=(--dataset Shenzhen --shenzhen-value-dir "${SHENZHEN_VALUE_DIR}" --shenzhen-station-csv "${SHENZHEN_STATION_CSV}")
    ;;

  *)
    echo "Unsupported dataset: ${DATASET}. Choose one of: UScoast|NDBC|Tehiku|Shenzhen" >&2
    exit 1
    ;;
esac

mkdir -p "${ROOT_OUT}"

# -------------------------------
# Sweep loops
# -------------------------------
for K in "${K_LIST[@]}"; do
  for NS in "${NUM_SUBDIV_LIST[@]}"; do
    for KING in "${KING_SELECT_LIST[@]}"; do
      for W in "${WEIGHT_LIST[@]}"; do
        for SEED in "${SEED_LIST[@]}"; do
          OUT_DIR="${ROOT_OUT}/K${K}_NS${NS}_KING${KING}_W${W}_seed${SEED}"
          mkdir -p "${OUT_DIR}"

          echo ">> Running ${DATASET} | K=${K} num_subdivisions=${NS} king_select=${KING} weight=${W} seed=${SEED}"
          set -x
          ${PY} main.py                 "${DS_ARG[@]}"                 --K "${K}"                 --num-subdivisions "${NS}"                 --king-select "${KING}"                 --weight-scalar "${W}"                 --seed "${SEED}"                 --out-dir "${OUT_DIR}"
          set +x
        done
      done
    done
  done
done

echo "All runs for ${DATASET} finished. Results under: ${ROOT_OUT}"
