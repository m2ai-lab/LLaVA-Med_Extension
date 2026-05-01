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


# # # NOW USE THE RIGHT PYTHON
# # python -c "import torch; print('torch OK')"
# # python -c "import llava; print('llava OK')"

# ###############################################################
# ############## FOR BLANK IMAGE REFERENCE ####################
# ###############################################################

# $DYNAMIC_PYTHON -m llava.eval.model_vqa \
#     --conv-mode mistral_instruct \
#     --model-path ./llava-med-v1.5-mistral-7b \
#     --question-file /scratch/group/CX000019_DS1/vlm-brain-mri/catherine/QApairs/LLaVA/question_blacked.jsonl \
#     --answers-file /scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/LLaVA/predicted_blank_results.csv \
#     --image-folder /scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/blank/ 

# ###############################################################
# ############## FOR FINALIZED PDGM QA PAIRS ####################
# ###############################################################

# # Tailor input format - for "finalized_ucsf_pdgm_pairs.csv"
# $DYNAMIC_PYTHON ../QApairs/LLaVA/match_question_format.py \
#   --input /scratch/group/CX000019_DS1/vlm-brain-mri/finalized_ucsf_pdgm_pairs.csv \
#   --output /scratch/group/CX000019_DS1/vlm-brain-mri/catherine/QApairs/LLaVA/question_finialized.jsonl \
#   --blacked /scratch/group/CX000019_DS1/vlm-brain-mri/catherine/QApairs/LLaVA/question_blacked.jsonl

# ## Run model
# $DYNAMIC_PYTHON -m llava.eval.model_vqa \
#     --conv-mode mistral_instruct \
#     --model-path ./llava-med-v1.5-mistral-7b \
#     --question-file /scratch/group/CX000019_DS1/vlm-brain-mri/catherine/QApairs/LLaVA/question_finialized.jsonl \
#     --answers-file /mnt/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/LLaVA/predicted_answer_finilized_results.csv \
#     --image-folder /scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/2D_slices
#     # --temperature 0.0


# ###############################################################
# ############## FOR RESHUFFLE UPDATED QA PAIRS ####################
# ###############################################################

# # Tailor input format - for "finalized_ucsf_pdgm_pairs.csv"
# $DYNAMIC_PYTHON ../QApairs/LLaVA/match_question_format.py \
#   --input /scratch/group/CX000019_DS1/vlm-brain-mri/reshuffled_updated_ucsf_pdgm_pairs.csv \
#   --output /scratch/group/CX000019_DS1/vlm-brain-mri/catherine/QApairs/LLaVA/question_reshuffled_updated.jsonl \
#   --blacked /scratch/group/CX000019_DS1/vlm-brain-mri/catherine/QApairs/LLaVA/question_blacked.jsonl

# ## Run model
# $DYNAMIC_PYTHON -m llava.eval.model_vqa \
#     --conv-mode mistral_instruct \
#     --model-path ./llava-med-v1.5-mistral-7b \
#     --question-file /scratch/group/CX000019_DS1/vlm-brain-mri/catherine/QApairs/LLaVA/question_reshuffled_updated.jsonl \
#     --answers-file /mnt/scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/LLaVA/predicted_answer_reshuffled_updated_results.csv \
#     --image-folder /scratch/group/CX000019_DS1/vlm-brain-mri/QApairs/format_dataset/2D_slices
#     # --temperature 0.0


###############################################################################
# PATHS
###############################################################################
ROOT="/scratch/group/CX000019_DS1/vlm-brain-mri"
QROOT="$ROOT/catherine/QApairs/LLaVA"
AROOT="$ROOT/catherine/QApairs/LLaVA"
IMGROOT="$ROOT/QApairs/format_dataset"

BLANK_Q="$QROOT/question_blacked.jsonl"
BLANK_A="$AROOT/predicted_blank_results.csv"
BLANK_IMG="$IMGROOT/blank"

FINAL_CSV="$ROOT/new_refined_ucsf_pdgm_pairs.csv"
FINAL_Q="$QROOT/question_refined.jsonl"
FINAL_A="$AROOT/predicted_answer_refined_results.csv"
FINAL_IMG="$IMGROOT/2D_slices"

# RESHUFFLE_CSV="$ROOT/reshuffled_finalized_ucsf_pdgm_pairs.csv"
# RESHUFFLE_Q="$QROOT/question_reshuffled_updated.jsonl"
# RESHUFFLE_A="$AROOT/predicted_answer_reshuffled_updated_results.csv"
# RESHUFFLE_IMG="$IMGROOT/2D_slices"

###############################################################################
# HELPER FUNCTIONS
###############################################################################

print_header () {
    local title="$1"
    echo
    echo "============================================================"
    echo "$title"
    echo "============================================================"
}

inspect_file () {
    local label="$1"
    local file="$2"

    if [ -f "$file" ]; then
        echo "Line count:"
        wc -l "$file"
        echo "First 3 lines:"
        head -n 3 "$file" || true
    else
        echo "Exists: NO"
    fi

    echo "--------------------"
}

inspect_jsonl () {
    local label="$1"
    local file="$2"

    echo "[JSONL CHECK] $label"

    if [ ! -f "$file" ]; then
        echo "File not found"
        echo "--------------------"
        return
    fi

    "$DYNAMIC_PYTHON" - <<PY
import json
path = r"""$file"""
good = 0
bad = 0
bad_lines = []

with open(path, "r") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            good += 1
        except Exception as e:
            bad += 1
            bad_lines.append((i, str(e)))

print("Valid JSON rows:", good)
print("Bad JSON rows:", bad)
if bad_lines:
    print("First bad rows:")
    for item in bad_lines[:5]:
        print(item)
PY

    echo "--------------------"
}

compare_counts () {
    local qfile="$1"
    local afile="$2"
    local label="$3"

    local qcount=0
    local acount=0

    [ -f "$qfile" ] && qcount=$(wc -l < "$qfile")
    [ -f "$afile" ] && acount=$(wc -l < "$afile")

    echo "--------------------"
    echo "[COUNT COMPARISON] $label"
    echo "Questions: $qcount"
    echo "Answers:   $acount"
    echo "Missing:   $((qcount - acount))"
    echo "--------------------"
}

check_nonempty_file () {
    local label="$1"
    local file="$2"

    if [ ! -f "$file" ]; then
        echo "ERROR: $label was not created: $file"
        exit 1
    fi

    local lines
    lines=$(wc -l < "$file")

    if [ "$lines" -eq 0 ]; then
        echo "ERROR: $label is empty: $file"
        exit 1
    fi

    echo "$label exists and is non-empty: $lines lines"
}

###############################################################################
# RUN QUESTION FILE FORMATTING
###############################################################################
print_header "BLOCK 0: QUESTION FILE FORMATTING"
inspect_file "FINALIZED CSV INPUT BEFORE FORMATTER" "$FINAL_CSV"
inspect_file "RESHUFFLED CSV INPUT BEFORE FORMATTER" "$RESHUFFLE_CSV"

"$DYNAMIC_PYTHON" ../QApairs/LLaVA/match_question_format.py \
    --input "$FINAL_CSV" \
    --output "$FINAL_Q" \
    --blacked "$BLANK_Q"

"$DYNAMIC_PYTHON" ../QApairs/LLaVA/match_question_format.py \
    --input "$RESHUFFLE_CSV" \
    --output "$RESHUFFLE_Q" \
    --blacked "$BLANK_Q"

###############################################################################
# (input check) BLANK IMAGE REFERENCE 
###############################################################################
print_header "BLOCK 1: (input check) BLANK IMAGE REFERENCE "
inspect_jsonl "BLANK QUESTION INPUT AFTER FORMATTING" "$BLANK_Q"

###############################################################################
# (input check & formatting) FINALIZED PDGM QA PAIRS
###############################################################################
print_header "BLOCK 2: (input check & formatting) FINALIZED PDGM QA PAIRS"

check_nonempty_file "FINALIZED GENERATED JSONL" "$FINAL_Q"
inspect_file "FINALIZED GENERATED JSONL AFTER FORMATTER" "$FINAL_Q"
inspect_jsonl "FINALIZED GENERATED JSONL AFTER FORMATTER" "$FINAL_Q"

###############################################################################
# (input check & formatting) RESHUFFLED UPDATED QA PAIRS
###############################################################################
# print_header "BLOCK 3: (input check & formatting) RESHUFFLED UPDATED QA PAIRS"

# check_nonempty_file "RESHUFFLED GENERATED JSONL" "$RESHUFFLE_Q"
# inspect_file "RESHUFFLED GENERATED JSONL AFTER FORMATTER" "$RESHUFFLE_Q"
# inspect_jsonl "RESHUFFLED GENERATED JSONL AFTER FORMATTER" "$RESHUFFLE_Q"



###############################################################################
# (RUN) BLANK IMAGE 
###############################################################################
print_header "BLOCK 4: (RUN) BLANK IMAGE "

"$DYNAMIC_PYTHON" -m llava.eval.model_vqa \
    --conv-mode mistral_instruct \
    --model-path ./llava-med-v1.5-mistral-7b \
    --question-file "$BLANK_Q" \
    --answers-file "$BLANK_A" \
    --image-folder "$BLANK_IMG"

inspect_file "BLANK ANSWER OUTPUT AFTER RUN" "$BLANK_A"

###############################################################################
# (RUN) FINALIZED PDGM QA PAIRS
###############################################################################

print_header "BLOCK 5: (RUN) FINALIZED PDGM QA PAIRS"

"$DYNAMIC_PYTHON" -m llava.eval.model_vqa \
    --conv-mode mistral_instruct \
    --model-path ./llava-med-v1.5-mistral-7b \
    --question-file "$FINAL_Q" \
    --answers-file "$FINAL_A" \
    --image-folder "$FINAL_IMG"

inspect_file "FINALIZED ANSWER OUTPUT AFTER RUN" "$FINAL_A"

###############################################################################
# (run) RESHUFFLED UPDATED QA PAIRS
###############################################################################
# print_header "BLOCK 6:  (run) RESHUFFLED UPDATED QA PAIRS"

# "$DYNAMIC_PYTHON" -m llava.eval.model_vqa \
#     --conv-mode mistral_instruct \
#     --model-path ./llava-med-v1.5-mistral-7b \
#     --question-file "$RESHUFFLE_Q" \
#     --answers-file "$RESHUFFLE_A" \
#     --image-folder "$RESHUFFLE_IMG"

# inspect_file "RESHUFFLED ANSWER OUTPUT AFTER RUN" "$RESHUFFLE_A"

###############################################################################
# FINAL SUMMARY
###############################################################################
print_header "FINAL SUMMARY"
compare_counts "$BLANK_Q" "$BLANK_A" "BLANK"
compare_counts "$FINAL_Q" "$FINAL_A" "FINALIZED"
# compare_counts "$RESHUFFLE_Q" "$RESHUFFLE_A" "RESHUFFLED_UPDATED"

echo
echo "All blocks completed."
echo "End time: $(date)"

