import argparse, json, os, random, torch, re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
random.seed(42)

def get_args():
    p = argparse.ArgumentParser('Dialogue generator')
    p.add_argument('--llm', required=True, type=str,
                   help='HF model id for causal LM chat model')
    p.add_argument('--llm_name', required=True, type=str,
                   help='Name of the LLM')
    p.add_argument('--pairing', required=True,
                   choices=['positive','negative','mixed','opposite', 'original'],
                   help='Which profile bucket(s) to pair')
    p.add_argument('--K', type=int, default=5, help='Sentences per profile')
    p.add_argument('--N', type=int, default=3000, help='Dialogues to create if cache missing')
    return p.parse_args()

def user_pairing(profile):
    random_indices = random.sample(range(len(profile)), 2)
    users = [profile[i] for i in random_indices]
    user1 = " ".join(users[0])
    user2 = " ".join(users[1])
    paired_users = f'''User 1: {user1}
    User 2: {user2}'''
    return paired_users

def opposite_user_pairing(p, n):
    positive = random.choice(p)
    negative = random.choice(n)
    if random.random() < 0.5:
        user1 = random.choice(positive)
        user2 = random.choice(negative)
    else:
        user1 = random.choice(negative)  
        user2 = random.choice(positive)  
    paired_users = f'''User 1: {user1}
    User 2: {user2}'''
    
    return paired_users



def dialogue_generation(args):
    model = AutoModelForCausalLM.from_pretrained(
    args.llm,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    
    users = []
    dialogues = []
    
    pairing_path = f'{args.pairing}_users_{args.K}.json'
    if os.path.exists(pairing_path):
        with open(pairing_path, encoding="utf-8") as f:
            users = json.load(f)
        for pairing in tqdm(users):
            messages = [
                {"role": "system",
                 "content": pairing
                },
                {"role": "user", 
                 "content": "Generate a dialogue between User 1 and User 2."
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
                max_new_tokens=4096,
                do_sample=False
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]        
            dialogues.append(response)
            
    else:    
        while len(users) < args.N: 
            if args.pairing == "opposite":
                with open(f'positive_profiles_{args.K}.json', 'r', encoding='utf-8') as f:
                    p = json.load(f)
                with open(f'negative_profiles_{args.K}.json', 'r', encoding='utf-8') as f:
                    n = json.load(f)
                pairing = opposite_user_pairing(p, n)
            else:
                with open(f'{args.pairing}_profiles_{args.K}.json', 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                pairing = user_pairing(profiles)
        
            if pairing in users:
                continue
        
            messages = [
                {"role": "system",
                 "content": pairing
                },
                {"role": "user", 
                 "content": "Generate a dialogue between User 1 and User 2."
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
                max_new_tokens=4096,
                do_sample=False
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
            users.append(pairing)
            dialogues.append(response)
    
        with open(f'{args.pairing}_users_{args.K}.json', 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=4)
            
    with open(f'{args.llm_name}_{args.pairing}_dialogues_{args.K}_joint.json', 'w', encoding='utf-8') as f:
            json.dump(dialogues, f, ensure_ascii=False, indent=4)
            
if __name__ == '__main__':
    dialogue_generation(get_args())