set -ex

export CUDA_VISIBLE_DEVICES=0

cd src/evaluation

NAME="llama"                 # Short name of the model
PAIR="original"              # original | positive | negative | mixed | opposite
K=5                          # personas per profile
STRATEGY="tb"             # joint | tb(turn-based)

python C2_eval.py \
  --llm_name $NAME \
  --pairing $PAIR \
  --K $K \
  --strategy $STRATEGY