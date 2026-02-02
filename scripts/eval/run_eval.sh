#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}"

# ============================================
# Configuration: Select task number(s)
# ============================================
# 1: hellaswag
# 2: lambada_openai
# 3: winogrande
# 4: openbookqa
# 5: arc_easy
# 6: arc_challenge
# 7: piqa
# 8: social_iqa
# 9: storycloze_2018
# 10: lm
# 11: harness
# ============================================
# 예시: TASK_NUMS="1" 또는 TASK_NUMS="1,3,5" (여러 개 선택 가능)
TASK_NUMS="1"  # ← 여기 숫자만 수정하세요! (쉼표로 여러 개 선택 가능)

# ============================================
# Configuration: Select model checkpoint(s)
# ============================================
# 평가할 체크포인트 경로를 지정하세요 (여러 개 가능)
# 예시:
#   MODELS=("/path/to/model1" "/path/to/model2")
#   또는
#   MODELS=("/home/jiwonyoon/data1/projects/MiniPLM/results/pretrain/qwen_200M/.../100000")
# ============================================
MODELS=(
  # "/home/jiwonyoon/data1/projects/MiniPLM/results/offline_kd/sparse_kd/qwen_200M/t100K-w2K-bs32-lr0.0006cosine6e-05-G2-N8-NN1-scr/offline-topk-a0.5/100000"
  # "/home/jiwonyoon/data1/projects/MiniPLM/results/offline_kd/sparse_kd/qwen_500M/t100K-w2K-bs32-lr0.0003cosine3e-05-G2-N8-NN1-scr/offline-topk-a0.5/100000"
  # "/home/jiwonyoon/data1/projects/MiniPLM/results/offline_kd/topk/qwen_200M/t100K-w2K-bs32-lr0.0006cosine6e-05-G2-N8-NN1-scr/offline-topk-a0.5/100000"
  # "/home/jiwonyoon/data1/projects/MiniPLM/results/pretrain/qwen_1.2B/t100K-w2K-bs32-lr0.00025cosine2.5e-05-G2-N8-NN1-scr/100000"
  # "/home/jiwonyoon/data1/projects/MiniPLM/results/pretrain/qwen_200M/t100K-w2K-bs64-lr0.0006cosine6e-05-G1-N8-NN1-scr/100000"
  # "/home/jiwonyoon/data1/projects/MiniPLM/results/pretrain/qwen_500M/t100K-w2K-bs32-lr0.0006cosine6e-05-G2-N8-NN1-scr/100000"
  # "/home/jiwonyoon/data1/projects/MiniPLM/results/vanilla_kd/miniplm_refined_corpus/qwen_200M/t100K-w2K-bs16-lr0.0006cosine6e-05-G4-N8-NN1-scr/7B-kd0.5/100000"
  "/home/jiwonyoon/data1/projects/MiniPLM/results/pretrain/miniplm_refined_corpus/qwen_200M-2stage-sft/t100K-w2K-bs64-lr0.0006cosine6e-05-G1-N8-NN1/90000"
)

# ============================================
# Function: Map number to task name
# ============================================
num_to_task() {
  local num=$1
  case $num in
    1) echo "hellaswag" ;;
    2) echo "lambada_openai" ;;
    3) echo "winogrande" ;;
    4) echo "openbookqa" ;;
    5) echo "arc_easy" ;;
    6) echo "arc_challenge" ;;
    7) echo "piqa" ;;
    8) echo "social_iqa" ;;
    9) echo "storycloze_2018" ;;
    10) echo "lm" ;;
    11) echo "harness" ;;
    *)
      echo "❌ Invalid task number: $num (must be 1-11)" >&2
      return 1
      ;;
  esac
}

# ============================================
# Function: Get script path from task name
# ============================================
get_script_path() {
  local task=$1
  if [[ "$task" == "lm" ]]; then
    echo "${SCRIPT_DIR}/lm.sh"
  elif [[ "$task" == "harness" ]]; then
    echo "${SCRIPT_DIR}/harness.sh"
  else
    echo "${SCRIPT_DIR}/${task}/eval_${task}.sh"
  fi
}

# ============================================
# Parse and run tasks
# ============================================
IFS=',' read -ra TASK_ARRAY <<< "$TASK_NUMS"
TOTAL_TASKS=${#TASK_ARRAY[@]}

echo "=========================================="
echo "  Running $TOTAL_TASKS evaluation task(s)"
echo "=========================================="
echo ""

for i in "${!TASK_ARRAY[@]}"; do
  TASK_NUM=$(echo "${TASK_ARRAY[$i]}" | xargs)  # trim whitespace
  TASK=$(num_to_task "$TASK_NUM")
  
  if [[ $? -ne 0 ]]; then
    exit 1
  fi
  
  SCRIPT_PATH=$(get_script_path "$TASK")
  
  # Check if script exists
  if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "❌ Script not found: $SCRIPT_PATH"
    exit 1
  fi
  
  # Make script executable
  chmod +x "$SCRIPT_PATH"
  
  # Create temporary script with MODELS replaced (in same directory as original)
  TEMP_SCRIPT="${SCRIPT_PATH}.tmp"
  trap "rm -f $TEMP_SCRIPT" EXIT
  
  # Find the line numbers of MODELS array
  START_LINE=$(grep -n "^MODELS=(" "$SCRIPT_PATH" | cut -d: -f1 | head -1)
  
  if [[ -n "$START_LINE" ]]; then
    # Find the closing line (next line with just ")" after MODELS=()
    END_LINE=$(awk -v start="$START_LINE" 'NR > start && /^\)$/ {print NR; exit}' "$SCRIPT_PATH")
    
    if [[ -n "$END_LINE" ]]; then
      # Create temp script with replaced MODELS
      {
        head -n $((START_LINE - 1)) "$SCRIPT_PATH"
        echo "MODELS=("
        for model in "${MODELS[@]}"; do
          echo "  \"$model\""
        done
        echo ")"
        tail -n +$((END_LINE + 1)) "$SCRIPT_PATH"
      } > "$TEMP_SCRIPT"
    else
      # Fallback: just copy the script
      cp "$SCRIPT_PATH" "$TEMP_SCRIPT"
    fi
  else
    # No MODELS array found, just copy
    cp "$SCRIPT_PATH" "$TEMP_SCRIPT"
  fi
  
  chmod +x "$TEMP_SCRIPT"
  
  # Run the selected script
  echo "=========================================="
  echo "[$((i+1))/$TOTAL_TASKS] Running: $TASK (task #$TASK_NUM)"
  echo "📄 Script: $SCRIPT_PATH"
  echo "📦 Models: ${#MODELS[@]} checkpoint(s)"
  echo "=========================================="
  echo ""
  
  # Run script (not exec, so we can continue to next task)
  "$TEMP_SCRIPT" "$@"
  
  EXIT_CODE=$?
  if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "❌ Task $TASK failed with exit code $EXIT_CODE"
    echo "Stopping execution."
    exit $EXIT_CODE
  fi
  
  echo ""
  echo "✅ Task $TASK completed successfully"
  echo ""
done

echo "=========================================="
echo "  All tasks completed successfully!"
echo "=========================================="
