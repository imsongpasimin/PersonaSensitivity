#!/usr/bin/env python
# coding: utf-8

import re
import json
import re
import torch
import numpy as np
import openai
from openai import OpenAI
import argparse


def get_args():
    p = argparse.ArgumentParser('Persona-dialogue probability evaluator')
    p.add_argument('--llm_name', required=True, help='Prefix used in dialogue filenames')
    p.add_argument('--pairing', required=True, choices=['original','positive','negative','mixed','opposite'])
    p.add_argument('--K', type=int, default=5, help='Sentences per profile')
    p.add_argument('--strategy', default='joint', choices=['tb','joint'], help='Turn‑based(tb) or joint')
    p.add_argument('--evaluator', default='gpt-4o', help='LLM Evaluator')
    p.add_argument('--eval_type', choices=['consistency','coherence'])
    return p.parse_args()


def prompt(persona, dialogue, eval_type):    
    if eval_type == "consistency":
        instruction = f'''You will be given two of user persona descriptions and a dialogue between these users.

        Your task is to rate the dialogue on one metric.

        Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.


        Evaluation Criteria:

        Consistency (1-5) - the factual alignment between the Persona descriptions and the utterances in the dialogue. A factually consistent utterance contains only statements that are entailed by the source persona descriptions. Annotators were also asked to penalize dialogue that contained utterances contradicted by Persona descriptions.

        - 1: Dialogue misses all personas in Persona descriptions, or contains many utterances contradicted by Persona descriptions.
        - 2: Dialogue misses three or four personas in Persona descriptions, or contains several utterances contradicted by Persona descriptions.
        - 3: Dialogue misses two personas in Persona descriptions, or contains few utterances contradicted by Persona description.
        - 4: Dialogue misses one persona in Persona descriptions.
        - 5: Dialogue reflects all of Persona descriptions perfectly.

        Evaluation Steps:

        1. Read the persona descriptions carefully and identify the main facts and details it presents.
        2. Read the dialogue and compare it to the descriptions. Check if the dialogue contains any utterances contradicted by given descriptions.
        3. Assign a score for consistency based on the Evaluation Criteria. BE STRICT IN YOUR EVALUATION.
        4. Justify the rating by referring to specific aspects of the conversation that demonstrate its coherence or lack thereof.

        Example:


        Persona descriptions:

        {persona}


        Dialogue:

        {dialogue}


        Evaluation Form (scores ONLY):

        - Consistency: '''
        
    elif eval_type == "coherence":
        instruction = f'''You will be given one conversation between two users.

        Your task is to rate the conversation on one metric. Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

        Evaluation Criteria:
    
        Coherence (1-5) - the collective quality of all utterances. The conversation should be well-structured and well-organized. The conversation should not just be a heap of related information, but should build from utterance to a coherent body of conversation about a topic and previous context."
     
        - 1: All utterances in dialogue are unrelated to each other, or the dialogue contains many utterances contradicted to the previous context.
        - 2: Dialogue contains some utterances contradicted to the previous context.
        - 3: Dialogue contains some utterances unrelated to the previous context.
        - 4: Dialogue is somewhat fluent, but the flow of the topic is not smooth.
        - 5: Dialgoue is perfectly fluent and well-organized.

        Evaluation Steps:
    
        1. Read and understand the given conversation.
        2. Evaluate the conversation based on the coherence of the utterances.
        3. Rate the conversation on a scale of 1 to 5 based on Evaluation Criteria. BE STRICT IN YOUR EVALUATION.
        4. Justify the rating by referring to specific aspects of the conversation that demonstrate its coherence or lack thereof.

        Example:
    
        Conversation: {dialogue}

        Evaluation Form (scores ONLY):
        - Coherence: '''
        
    return instruction


# In[37]:

def main():
    args = get_args()
    
    with open(f'{args.llm_name}_{args.pairing}_users_{args.K}_{args.strategy}_final.json', 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    with open(f'{args.llm_name}_{args.pairing}_dialogues_{args.K}_{args.strategy}_final.json', 'r', encoding='utf-8') as f:
        dialogue = json.load(f)
    
    # Use Batch API to reduce costs
    batch_files = []
    for i in range(len(profiles)):
        task = {
            "custom_id": f"task-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": args.evaluator,
                "temperature": 0,
                "messages": [
                    {
                    "role": "system",
                    "content": f"You are a {args.eval_type} evaluator."
                    },
                    {
                    "role": "user",
                    "content": prompt(profiles[i], dialogue[i], args.eval_type)
                    }
                ],
            }
        }
        batch_files.append(task)

    file_name = f"G-eval_{args.eval_type}_input.jsonl"

    with open(file_name, 'w') as file:
        for obj in batch_files:
            file.write(json.dumps(obj) + '\n')

    client = OpenAI(api_key="<YOUR_API_KEY>")

    batch_input_file = client.files.create(
      file=open(file_name, "rb"),
      purpose="batch"
    )


    batch_job = client.batches.create(
      input_file_id=batch_input_file.id,
      endpoint="/v1/chat/completions",
      completion_window="24h"
    )

    # File ID to download (obtained after the Batch API call)
    file_id = '<output_file_id>'

    # Download the file (returned as binary data)
    downloaded_file = client.files.retrieve_content(file_id)

    if isinstance(downloaded_file, str):
        file_content = downloaded_file.encode('utf-8')
    # Save the downloaded file locally (e.g., output.jsonl)

    output_file = f"G-eval_{args.eval_type}_input.jsonl"
    with open(output_file, "wb") as f:
        f.write(file_content)

    # Load the saved file (for JSONL format)
    with open(output_file, "r", encoding="utf-8") as f:
        # Each line should contain a single JSON object
        datas = [json.loads(line) for line in f if line.strip()]
        
        
if __name__ == '__main__':
    main()