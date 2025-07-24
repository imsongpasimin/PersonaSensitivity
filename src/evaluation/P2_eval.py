#!/usr/bin/env python
# coding: utf-8

import re
import json
import re
import torch
import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math
import statistics
import argparse



def get_args():
    p = argparse.ArgumentParser('Persona-dialogue probability evaluator')
    p.add_argument('--llm_name', required=True, help='Prefix used in dialogue filenames')
    p.add_argument('--pairing', required=True, choices=['original','positive','negative','mixed','opposite'])
    p.add_argument('--K', type=int, default=5, help='Sentences per profile')
    p.add_argument('--strategy', default='joint', choices=['tb','joint'], help='Turn‑based(tb) or joint')
    p.add_argument('--perplexity_model', default='gpt2-large', help='Language model to calculate perplexity')
    return p.parse_args()


def Perplexity(P, D, tokenizer, model):
    persona_context = P.replace("User 1:", "User 1 Persona:").replace("    User 2:", "User 2 Persona:") + "\n\n"

    # Calculate P(D)
    inputs = tokenizer.encode(D, return_tensors='pt')
    inputs = inputs.to('cuda')
    
    # Calculate P(D|U1,U2)
    conditioned_input = persona_context + D
    inputs_conditioned = tokenizer.encode(conditioned_input, return_tensors='pt')
    token_length = inputs_conditioned.size(1)
    if token_length > 1024:              # Filter out the samples exceeding max token lengths
        return (None, None, None)
    
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        log_likelihood = loss.item()
        P_D = math.exp(log_likelihood)

    # Generate labels for masking
    labels = inputs_conditioned.clone()
    persona_length = tokenizer.encode(persona_context, return_tensors='pt').size(1)
    labels[0, :persona_length] = -100  # Ignore Persona Tokens
    
    inputs_conditioned = inputs_conditioned.to('cuda')
    labels = labels.to('cuda')

    with torch.no_grad():
        outputs = model(inputs_conditioned, labels=labels)
        loss_conditioned = outputs.loss
        log_likelihood_conditioned = loss_conditioned.item()
        P_D_given_U1_U2 = math.exp(log_likelihood_conditioned)


    # Calculate Perplexity gap
    delta = P_D_given_U1_U2 - P_D
        
    return P_D, P_D_given_U1_U2, delta


def stat(scores):
    mean = sum(scores) / len(scores)
    std_dev = statistics.stdev(scores)

    return mean, std_dev,len(scores)

def main():
    args = get_args()

    tokenizer = GPT2Tokenizer.from_pretrained(args.perplexity_model)
    model = GPT2LMHeadModel.from_pretrained(args.perplexity_model)
    model.to('cuda')
    model.eval()


    with open(f'{args.llm_name}_{args.pairing}_users_{args.K}_{args.strategy}_final.json', 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    with open(f'{args.llm_name}_{args.pairing}_dialogues_{args.K}_{args.strategy}_final.json', 'r', encoding='utf-8') as f:
        dialogue = json.load(f)


    p_gap = []
    perp = []
    for i in tqdm(range(len(profiles))):
        p_d, p_d_p, delta = Perplexity(profiles[i], dialogue[i], tokenizer, model)
        if p_d:
            p_gap.append(delta)
            perp.append(p_d)

    p_gap_mean, p_gap_std, p_gap_len = stat(p_gap)
    perp_mean, perp_std, perp_len = stat(perp)

    print("P_gap: ", p_gap_mean)
    print("Perplexity: ", perp_mean)
    
if __name__ == '__main__':
    main()