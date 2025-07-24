set -ex

export CUDA_VISIBLE_DEVICES=0

cd src/evaluation

NAME="llama"                 # Short name of the model
PAIR="original"              # original | positive | negative | mixed | opposite
K=5                          # personas per profile
STRATEGY="tb"             # joint | tb(turn-based)
MODEL="gpt2-large"           # Backbone Language Model to calculate perplexity (GPT2 Series)

python P2_eval.py \
  --llm_name $NAME \
  --pairing $PAIR \
  --K $K \
  --strategy $STRATEGY \
  --perplexity_model $MODEL