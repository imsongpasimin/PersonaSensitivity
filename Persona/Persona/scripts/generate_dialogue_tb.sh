set -ex

export CUDA_VISIBLE_DEVICES=0

cd src/dialogue_generation

LLM="meta-llama/Llama-3.2-3B-Instruct"   # You can use various models
NAME="llama"                 # Short name of the model
SENT="distilbert-base-uncased-finetuned-sst-2-english"
PAIR="original"              
K=5                          # personas per profile
ORDER="WGH"                  # ASC | DSC | WGH | default
P="sentiment"                # If sentiment, add sentiment-aware prompt

python dialogue_generation_tb.py \
  --llm $LLM \
  --llm_name $NAME \
  --sent_model $SENT \
  --pairing $PAIR \
  --K $K \
  --order $ORDER \
  --prompt $P \