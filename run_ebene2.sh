#!/usr/bin/env bash
set -euo pipefail

# 1. RDNA3 & ROCm Konfiguration
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0

# 2. VENV aktivieren
source ~/8th_unit/venv_sovereign/bin/activate

MODEL="/home/predator/models/Qwen2.5-Coder-7B-Instruct"
TASKS="data/pressure_tasks.jsonl"
OUT_DIR="runs/q7b_oneshot"
LAYER=10

echo "=========================================================="
echo " STARTE EBENE 2: AUDIT & SUMMARY (LAYER $LAYER)"
echo " Branches: A (Baseline), B (Ablation), C (Steer), D (Noise)"
echo "=========================================================="

python pipeline.py \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --out "$OUT_DIR" \
  --device cuda \
  --stages probe,calibrate,audit,summary \
  --layer "$LAYER" \
  --force-audit \
  --samples 4

echo ""
echo "=========================================================="
echo " EBENE 2 ABGESCHLOSSEN - ERGEBNIS-SUMMARY:"
echo "=========================================================="
cat "$OUT_DIR/summary.json"
