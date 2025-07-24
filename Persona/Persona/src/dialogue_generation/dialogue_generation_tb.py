import argparse, json, random, torch, re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
random.seed(42)

def get_args():
    p = argparse.ArgumentParser('Dialogue generator')
    p.add_argument('--llm', required=True, type=str,
                   help='HF model id for causal LM chat model')
    p.add_argument('--llm_name', required=True, type=str,
                   help='Name of the LLM')
    p.add_argument("--sent_model", required=True, type=str, 
                   help="2-way sentiment classifier (0=neg,1=pos)")
    p.add_argument('--pairing', required=True,
                   default='original',
                   help='Which profile bucket(s) to pair')
    p.add_argument('--K', type=int, default=5, help='Sentences per profile')
    p.add_argument('--order', type=str, choices=['ASC', 'DSC', 'WGH', 'default'], help='Profile ordering strategy')
    p.add_argument('--prompt', type=str, default='sentiment', help='Sentiment-aware prompting')
    return p.parse_args()


def cs_ordering(classifier, sentences, order_type):
    results = []
    for sentence in sentences:
        output = classifier(sentence)[0]
        if order_type == 'WGH':
            numeric_label = 0.1 if output['label'] == 'POSITIVE' else 0.0
            custom_metric = abs(output['score'] - 0.5) + numeric_label
        else:
            numeric_label = 1.0 if output['label'] == 'POSITIVE' else 0.0
            custom_metric = abs(numeric_label + output['score'] - 1)
        results.append({'sentence': sentence, 'custom_metric': custom_metric})
    reverse = order_type == 'DSC'
    sorted_results = sorted(results, key=lambda x: x['custom_metric'], reverse=reverse)
    return [item['sentence'] for item in sorted_results]


def ordering_profiles(profiles, classifier, order_type):
    if order_type not in ['ASC', 'DSC', 'WGH']:
        return profiles
    
    ordered_profiles = []
    for c in tqdm(profiles):
        user_personas = c.split('\n')
        for user in user_personas:
            if user.startswith('User 1:'):
                user_1_persona = re.split(r'(?<=\.) ', user.replace('User 1: ', ''))
            elif user.startswith('    User 2:'):
                user_2_persona = re.split(r'(?<=\.) ', user.split('<|eot_id|>')[0].replace('User 2: ', ''))
        user_1_persona = [p.strip() for p in user_1_persona if p]
        user_2_persona = [p.strip() for p in user_2_persona if p]
        ordered_profile1 = cs_ordering(classifier, user_1_persona, order_type)
        ordered_profile2 = cs_ordering(classifier, user_2_persona, order_type)
        joined_profile1 = ' '.join(ordered_profile1)
        joined_profile2 = ' '.join(ordered_profile2)
        ordered_profile = "User 1: " + joined_profile1 + "\n    User 2:" + joined_profile2
        ordered_profiles.append(ordered_profile)                        
    return ordered_profiles



def dialogue_generation(args):    
    model = AutoModelForCausalLM.from_pretrained(
    args.llm,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    
    pairing_path = f'{args.pairing}_users_{args.K}.json'
    with open(pairing_path, encoding="utf-8") as f:
        original = json.load(f)
    
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline('sentiment-analysis', model=args.sent_model, device=device)
    users = ordering_profiles(original, classifier, args.order)
    
    # Sample the number of turns
    data = {14: 159, 16: 131, 18: 93, 12: 93, 15: 83, 13: 70, 17: 59, 20: 48, 19: 44, 11: 34, 10: 34, 22: 30, 21: 21, 23: 12, 24: 9, 26: 9, 27: 7, 25: 6, 28: 5, 30: 4, 8: 2, 9: 2, 7: 1, 43: 1, 32: 1, 29: 1}
    keys = list(data.keys())
    weights = list(data.values())
    # In[20]:
    samples_original = random.choices(keys, weights=weights, k=len(users))
    
    dialogues = []
    instruction_init = "Start a conversation with introducing yourself with part of given personas in 1~2 short sentences."
    instruction = "Generate the next line of dialogue based on the previous context in 1~2 short sentences."
    if args.prompt == "sentiment":
        sentiment_aware_prompt = "Please ensure that user's personas, especially negative or neutral personas, are well integrated into the dialogue and that the overall dialogue remains coherent."
        instruction_init += sentiment_aware_prompt
        instruction += sentiment_aware_prompt
        
    for i in tqdm(range(len(users))):
        cnt = 0
        original_user1 = users[i].split('\n    ')[0].strip()
        original_user2 = users[i].split('\n    ')[1].strip()
        context = ""
        while cnt <= samples_original[i]:
            context += "User 1: "
            if cnt == 0:            
                messages = [
                    {"role": "system",
                     "content": original_user1
                    },
                    {"role": "user", 
                     "content": "You are User 1. " + instruction_init
                     },
                ]
            
            else:            
                messages = [
                    {"role": "system",
                     "content": original_user1
                    },
                    {"role": "user", 
                     "content": "You are User 1. " + instruction + "\nDialogue context:\n" + context
                     },
                ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] + '\n'
            context += response 
            context += "User 2: "
            cnt += 1
        
            messages = [
                {"role": "system",
                 "content": original_user2
                },
                {"role": "user", 
                 "content": "You are User 2. " + instruction + "\nDialogue context:\n" + context
                 },
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                do_sample=False
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=128,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0] + "\n"
            context += response
            cnt += 1
        
        dialogues.append(context.strip())
            
    with open(f'{args.llm_name}_{args.pairing}_dialogues_{args.K}_tb.json', 'w', encoding='utf-8') as f:
        json.dump(dialogues, f, ensure_ascii=False, indent=4)
            
if __name__ == '__main__':
    dialogue_generation(get_args())