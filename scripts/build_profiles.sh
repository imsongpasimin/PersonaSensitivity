set -ex

export CUDA_VISIBLE_DEVICES=0

cd src/profile_synthesis

MODEL="distilbert-base-uncased-finetuned-sst-2-english"  # 0=neg,1=pos
BATCH=32    # batch size

python build_polarized_profiles.py \
  --sent_model $MODEL \
  --nli_model cross-encoder/nli-deberta-v3-large \
  --N 10000 \
  --K 5 \
  --batch_size $BATCH