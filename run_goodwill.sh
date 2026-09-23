#!/usr/bin/env bash
set -euo pipefail

# 1. RDNA3 & ROCm Isolation
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export HSA_ENABLE_SDMA=0

# 2. VENV laden
source ~/8th_unit/venv_sovereign/bin/activate

MODEL="/home/predator/models/Qwen2.5-Coder-7B-Instruct"
TASKS="data/pressure_tasks.jsonl"
OUT_DIR="runs/q7b_oneshot"

echo "=========================================================="
echo " 1. EXPERIMENT 0: ROLLOUTS (Mit Incentive-Druck)"
echo "=========================================================="
python pipeline.py \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --out "$OUT_DIR" \
  --rollouts 8 \
  --gen-batch 8 \
  --temperature 0.7 \
  --device cuda \
  --stages rollouts

echo ""
echo "[+] Rollouts abgeschlossen. Statusbericht:"
cat "$OUT_DIR/p1_report.json"
echo ""

echo "=========================================================="
echo " 2. MANUELLE STICHPROBE (Grader & Marker Check)"
echo "=========================================================="
read -p "Prüfe '$OUT_DIR/grader_review_sample.jsonl' und drücke ENTER..."

echo ""
echo "=========================================================="
echo " 3. EBENE 1: PROBE, MONITOR & PRÄFIX-VERGLEICH"
echo "=========================================================="
python pipeline.py \
  --model "$MODEL" \
  --tasks "$TASKS" \
  --out "$OUT_DIR" \
  --device cuda \
  --stages probe,calibrate,monitor \
  --layers 6:26:2

echo ""
echo "[+] Ebene 1 abgeschlossen. Ergebnis-Report:"
cat "$OUT_DIR/monitor_report.json"
