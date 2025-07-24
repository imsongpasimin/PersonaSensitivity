import argparse, json, re

def get_args():
    p = argparse.ArgumentParser('Dialogue filtering')
    p.add_argument('--llm', required=True, type=str, help='HF model id for causal LM chat model')
    p.add_argument('--llm_name', required=True, type=str, help='Short name of the LLM')
    p.add_argument('--pairing', required=True, choices=['positive','negative','mixed','opposite', 'original'],
                   help='Pairing bucket used to build dialogues')
    p.add_argument('--K', type=int, default=5, help='Sentences per profile')
    p.add_argument('--strategy', required=True, choices=['joint', 'tb'],
                   help='Dialogue generation strategy')
    return p.parse_args()

def parse_utterances(dialogue_text: str):
    lines = dialogue_text.splitlines()
    results = []

    pattern = re.compile(r'^(?:\*\*)?([^:\*]+)(?:\*\*)?:\s*(.*)')

    for line in lines:
        line = line.strip()
        if not line:
            continue  # 빈 줄은 건너뜀

        match = pattern.match(line)
        if match:
            speaker = match.group(1).strip()     # Speaker mark
            utterance = match.group(2).strip()   # Utterance
            results.append((speaker, utterance))

    return results


def filter_dialogue(dialogue_text: str) -> bool:
    """
    Filtering criteria:
    1) Discard any dialogue that contains four turns or fewer.
    2) Discard any dialogue where the speaker name is not“User 1" or“User 2” (e.g., names like“Steve",“Sarah", etc.).
    3) Discard any dialogue in which the same utterance (string) appears two or more times (Verbal samples).
    """
    matches = parse_utterances(dialogue_text)

    # (1)
    if len(matches) <= 4:
        return False

    # (2)
    for speaker, _ in matches:
        if speaker not in ["User 1", "User 2", "[Setting", "Setting"]:
            return False

    # (3)
    utterance_count = {}
    for _, utterance in matches:
        utterance_count[utterance] = utterance_count.get(utterance, 0) + 1
        if utterance_count[utterance] > 1:
            return False

    # Do not filter a dialogue unless it meets all three conditions simultaneously.
    return True


# In[12]:


def filtering(dialogues, persona):
    filtered_dialogue = []
    filtered_persona = []
    for i, d in enumerate(dialogues):
        if filter_dialogue(d):
            filtered_dialogue.append(d)
            filtered_persona.append(persona[i])
    
    return filtered_dialogue, filtered_persona


def main():
    args = get_args()

    with open(f'{args.llm_name}_{args.pairing}_dialogues_{args.K}_{args.strategy}.json', 'r', encoding='utf-8') as f:
        dialogues = json.load(f)
    with open(f'{args.pairing}_users_{args.K}.json', 'r', encoding='utf-8') as f:
        users = json.load(f)
    
    filltered_dialogues, filltered_users = filtering(dialogues, users)

    with open(f'{args.llm_name}_{args.pairing}_users_{args.K}_{args.strategy}_final.json', 'w', encoding='utf-8') as f:
        json.dump(filltered_users, f, ensure_ascii=False, indent=4)            
    with open(f'{args.llm_name}_{args.pairing}_dialogues_{args.K}_{args.strategy}_final.json', 'w', encoding='utf-8') as f:
        json.dump(filltered_dialogues, f, ensure_ascii=False, indent=4)
        
if __name__ == '__main__':
    main()