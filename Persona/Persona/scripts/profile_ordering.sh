set -ex

export CUDA_VISIBLE_DEVICES=0

cd src/profile_synthesis

MODEL="distilbert-base-uncased-finetuned-sst-2-english"  # sentiment model
ORDER="WGH"      # ASC | DSC | WGH

python profile_ordering.py \
  --sent_model $MODEL \
  --order $ORDER \