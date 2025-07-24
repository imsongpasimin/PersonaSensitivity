#!/usr/bin/env python
# coding: utf-8

# In[29]:

import argparse
import json
import re
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm


def get_args():
    p = argparse.ArgumentParser('Persona-dialogue consistency evaluator')
    p.add_argument('--llm_name', required=True, help='Prefix used in dialogue filenames')
    p.add_argument('--pairing', required=True, choices=['original','positive','negative','mixed','opposite'])
    p.add_argument('--K', type=int, default=5, help='Sentences per profile')
    p.add_argument('--strategy', default='joint', choices=['tb','joint'], help='Turn‑based(tb) or joint')
    return p.parse_args()

def consistency_eval(profiles, dialogue, tokenizer, model, args):
    persona1_o = []
    persona2_o = []
    for c in profiles:
        # Extract each speaker's personas
        user_personas = c.split('\n')
        for user in user_personas:
            if user.startswith('User 1:'):
                user_1_persona = re.split(r'(?<=\.) ', user.replace('User 1: ', ''))
            elif user.startswith('    User 2:'):
                user_2_persona = re.split(r'(?<=\.) ', user.split('<|eot_id|>')[0].replace('User 2: ', ''))
            
        user_1_persona = [p.strip() for p in user_1_persona if p]
        user_2_persona = [p.strip() for p in user_2_persona if p]
        persona1_o.append(user_1_persona)
        persona2_o.append(user_2_persona)



    utterance1_o = []
    utterance2_o = []
    for d in dialogue:
        # Extract each speaker's utterances
        utterances = d.split('\n') if args.strategy == "tb" else d.split('\n\n')
        user_1_utterance = []
        user_2_utterance = []

        for u in utterances:
            if u.startswith('User 1:'):
                user_1_utterance.append(u.replace('User 1: ', '').strip())
            elif u.startswith('User 2:'):
                user_2_utterance.append(u.replace('User 2: ', '').strip())
            elif u.startswith('**User 1:**'):
                user_1_utterance.append(u.replace('**User 1:** ', '').strip())
            elif u.startswith('**User 2:**'):
                user_2_utterance.append(u.replace('**User 2:** ', '').strip())

        utterance1_o.append(user_1_utterance)
        utterance2_o.append(user_2_utterance)
        
    print(len(persona1_o))
    print(len(persona2_o))
    print(len(utterance1_o))
    print(len(utterance2_o))
    
    C1 = []
    # Define the input sentences
    for i in tqdm(range(len(persona1_o))):
        d_c = []
        for premise in tqdm(persona1_o[i]):
            for hypothesis in utterance1_o[i]:
        
                # Tokenize the input sentences
                inputs = tokenizer(premise, hypothesis, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                # Perform inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_label_id = torch.argmax(logits, dim=-1).item()
                if predicted_label_id == 0:
                    d_c.append(1)
                elif predicted_label_id == 1:
                    d_c.append(0)
                else:
                    d_c.append(-1)    
        try:
            d_score = sum(d_c) / len(utterance1_o[i])
        except:
            d_score = None
        try:
            contd = d_c.count(-1) / (d_c.count(-1) + d_c.count(1)) * 100
        except:
            contd = None
        C1.append([d_score, contd])

    C2 = []
    # Define the input sentences
    for i in tqdm(range(len(persona2_o))):
        d_c = []
        for premise in tqdm(persona2_o[i]):
            for hypothesis in utterance2_o[i]:
        
                # Tokenize the input sentences
                inputs = tokenizer(premise, hypothesis, return_tensors='pt', padding='max_length', max_length=256, truncation=True)
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                # Perform inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    predicted_label_id = torch.argmax(logits, dim=-1).item()
                if predicted_label_id == 0:
                    d_c.append(1)
                elif predicted_label_id == 1:
                    d_c.append(0)
                else:
                    d_c.append(-1)      
        
        try:
            d_score = sum(d_c) / len(utterance2_o[i])
        except:
            d_score = None
        try:
            contd = d_c.count(-1) / (d_c.count(-1) + d_c.count(1)) * 100
        except:
            contd = None
        C2.append([d_score, contd])

    return C1, C2
        
def process(C1, C2):
    c_score_lst = [(C1[i][0] + C2[i][0]) / 2 for i in range(min(len(C1), len(C2))) if C1[i][0] is not None and C2[i][0] is not None]
    contd_lst = [(C1[i][1] + C2[i][1]) / 2 for i in range(min(len(C1), len(C2))) if C1[i][1] is not None and C2[i][1] is not None]  
    c_score = sum(c_score_lst) / len(c_score_lst)
    contd = sum(contd_lst) / len(contd_lst)
    return c_score, contd


def main():
    args = get_args()
    print("argument parsed!")
    
    # Load the trained model and tokenizer
    model_path = 'nli_model'
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path).to("cuda")

    # Set the model to evaluation mode
    model.eval()
    
    with open(f'{args.llm_name}_{args.pairing}_users_{args.K}_{args.strategy}_final.json', 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    with open(f'{args.llm_name}_{args.pairing}_dialogues_{args.K}_{args.strategy}_final.json', 'r', encoding='utf-8') as f:
        dialogue = json.load(f)
    
    C1, C2 = consistency_eval(profiles, dialogue, tokenizer, model, args)
    c_score, contd = process(C1, C2)
    print("C Score: ", c_score)
    print("Contd.: ", contd)
    
if __name__ == '__main__':
    main()