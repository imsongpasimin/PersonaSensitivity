set -ex

export CUDA_VISIBLE_DEVICES=0

cd src/dialogue_generation

LLM="meta-llama/Llama-3.1-8B-Instruct"   # You can use various models
NAME="llama"                 # Short name of the model
PAIR="original"              # positive | negative | mixed | opposite
K=5                          # personas per profile
STRATEGY="tb"             # joint | tb(turn-based)

python filtering.py \
  --llm $LLM \
  --llm_name $NAME \
  --pairing $PAIR \
  --K $K \
  --strategy $STRATEGY