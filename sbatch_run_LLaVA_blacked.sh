#!/bin/bash
#SBATCH --job-name=llava-med
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

#SBATCH --output=/scratch/group/CX000019_DS1/vlm-brain-mri/catherine/logs/LLaVA_test_%j.out
#SBATCH --error=/scratch/group/CX000019_DS1/vlm-brain-mri/catherine/logs/LLaVA_test_%j.err

# 1. Load the cluster's Conda module
module load miniforge3
module load cuda
cd /scratch/group/CX000019_DS1/vlm-brain-mri/catherine/LLaVA-Med

# 2. Prevent "Ghost" package leakage (The root of our earlier problems)
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# 3. DYNAMIC ENVIRONMENT DETECTION
# This command finds the path to 'medgemma' for whoever is running the script
ENV_PATH=$(conda env list | grep -E '^llava-med\s' | awk '{print $NF}')

if [ -z "$ENV_PATH" ]; then
    echo "ERROR: Environment 'llava-med' not found for user $USER."
    echo "Please create it using: conda create -n llava-med python=3.10"
    exit 1
fi

# Set the path to the actual python executable
DYNAMIC_PYTHON="${ENV_PATH}/bin/python"

echo "------------------------------------------------------------"
echo "Job ID:        $SLURM_JOB_ID"
echo "User:          $USER"
echo "Env Path:      $ENV_PATH"
echo "Using Python:  $DYNAMIC_PYTHON"
echo "Working dir:   $PWD"
echo "------------------------------------------------------------"


# # NOW USE THE RIGHT PYTHON
# python -c "import torch; print('torch OK')"
# python -c "import llava; print('llava OK')"

$DYNAMIC_PYTHON -m llava.eval.model_vqa \
    --conv-mode mistral_instruct \
    --model-path ./llava-med-v1.5-mistral-7b \
    --question-file /scratch/group/CX000019_DS1/vlm-brain-mri/catherine/QApairs/LLaVA/question_blacked.jsonl \
    --answers-file /mnt/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/LLaVA/predicted_blank_results.csv \
    --image-folder /scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/blank/ 
    # --temperature 0.0
