set -ex

export CUDA_VISIBLE_DEVICES=0

cd src/dialogue_generation

LLM="meta-llama/Llama-3.1-8B-Instruct"   # You can use various models
NAME="llama"                 # Short name of the model
PAIR="positive"              # positive | negative | mixed | opposite | original
K=5                           # personas per profile
N=3000                        # dialogues if cache missing

python dialogue_generation_joint.py \
  --llm $LLM \
  --llm_name $NAME \
  --pairing $PAIR \
  --K $K \
  --N $N