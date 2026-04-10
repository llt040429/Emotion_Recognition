#!/bin/bash

set -euo pipefail

# Usage:
#   bash run.sh <AUDIO_WAV_DIR> <WAVLM_MODEL_DIR>
# Example:
#   bash run.sh /data/CREMA-D/AudioWAV /data/models/wavlm-large

AUDIO_DIR="${1:-}"
MODEL_DIR="${2:-}"

if [[ -z "${AUDIO_DIR}" || -z "${MODEL_DIR}" ]]; then
  echo "Usage: bash run.sh <AUDIO_WAV_DIR> <WAVLM_MODEL_DIR>"
  exit 1
fi

if [[ ! -d "${AUDIO_DIR}" ]]; then
  echo "Audio directory not found: ${AUDIO_DIR}"
  exit 1
fi

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "Model directory not found: ${MODEL_DIR}"
  exit 1
fi

mkdir -p data/processed
mkdir -p outputs/wavlm_cremad

python -m pip install -r requirements.txt

python train.py \
  --audio-dir "${AUDIO_DIR}" \
  --model-dir "${MODEL_DIR}" \
  --processed-dir data/processed \
  --output-dir outputs/wavlm_cremad \
  --num-classes 4 \
  --batch-size 8 \
  --epochs 12 \
  --learning-rate 1e-5 \
  --num-workers 8 \
  --fp16 \
  --freeze-feature-encoder
