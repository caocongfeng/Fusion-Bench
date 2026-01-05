#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_h100
#SBATCH --job-name=fusion_train
#SBATCH -t 1-4:00:00

#SBATCH --array=0-2

# 关键点：不让 SLURM 接管 stdout/stderr
#SBATCH --output=/dev/null

source activate newest_trl

mkdir -p slurm_logs

# =====================
# Array job settings
# 7 ratios × 3 methods = 21
# =====================
RATIOS=(1)   # 2 0.5 3 "1/3" 4 0.25
METHODS=(Mix N_T T_N)

TASK_ID=${SLURM_ARRAY_TASK_ID}
RATIO_IDX=$(( TASK_ID / 3 ))
METHOD_IDX=$(( TASK_ID % 3 ))

RAW_RATIO=${RATIOS[$RATIO_IDX]}
COMB_METHOD=${METHODS[$METHOD_IDX]}

# 将 1/3 转为 float，供 argparse 使用
if [[ "$RAW_RATIO" == "1/3" ]]; then
  REASONING_RATIO=$(python - <<'PY'
print(1.0 / 3.0)
PY
)
else
  REASONING_RATIO="$RAW_RATIO"
fi

# 文件名安全化（避免 1/3）
SAFE_RATIO=${RAW_RATIO//\//_}

echo "================log======================"

LOG_FILE="slurm_logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${TASK_ID}_${COMB_METHOD}_ratio_${SAFE_RATIO}.out"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "======================================"


echo "======================================"
echo "JOB_NAME            = ${SLURM_JOB_NAME}"
echo "JOB_ID              = ${SLURM_JOB_ID}"
echo "ARRAY_TASK_ID       = ${TASK_ID}"
echo "combination_method  = ${COMB_METHOD}"
echo "raw_ratio           = ${RAW_RATIO}"
echo "ratio_float         = ${REASONING_RATIO}"
echo "CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES:-unset}"
echo "======================================"
# =====================
# parameters
# =====================
MODEL_NAME="Qwen/Qwen3-4B"
STANDARD_INDEX=1500
PROJECT_NAME="math_fusion_combine_H100_v0.3"

# =====================
# training
# =====================
python Fusion_train_LLM_v0.3_trl.py \
  --model_name "${MODEL_NAME}" \
  --standard_index "${STANDARD_INDEX}" \
  --combination_method "${COMB_METHOD}" \
  --reasoning_chat_percentage "${REASONING_RATIO}" \
  --project_name "${PROJECT_NAME}"