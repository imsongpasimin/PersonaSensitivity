set -ex

export CUDA_VISIBLE_DEVICES=0

cd src/evaluation

NAME="llama"                 # Short name of the model
PAIR="original"              # original | positive | negative | mixed | opposite
K=5                          # personas per profile
STRATEGY="joint"             # joint | tb(turn-based)
EVALUATOR="gpt-4o"           # LLM Evaluator (GPT Series)
TYPE="consistency"           # consistency | coherence


python G-eval.py \
  --llm_name $NAME \
  --pairing $PAIR \
  --K $K \
  --strategy $STRATEGY \
  --evaluator $EVALUATOR \ 
  --eval_type $TYPE