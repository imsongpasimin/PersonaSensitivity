import argparse
import random, json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import CrossEncoder
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent


def get_args():
    p = argparse.ArgumentParser("Polarized profile constructor")
    p.add_argument("--sent_model",     type=str, required=True,
                   help="2-way sentiment classifier (0=neg,1=pos)")
    p.add_argument("--nli_model",    type=str, default="cross-encoder/nli-deberta-v3-large",
                   help="NLI cross-encoder for contradiction check")
    p.add_argument("--N",             type=int, default=10000,
                   help="Number of profiles to generate per polarity")
    p.add_argument("--K",             type=int, default=5,
                   help="Number of persona sentences per profile")
    p.add_argument("--batch_size",   type=int, default=32,
                   help="Batch size for sentiment inference")
    return p.parse_args()



def negative_sentences(tokenizer, model, sentences, batch_size=32):
    results = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.Softmax(dim=1)(outputs.logits)
        preds = torch.argmax(probs, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        for j, p in enumerate(preds):
            neg_flags = (p.item() == 0 and max_probs[j].item() >= 0.99)
            results.append(neg_flags)
    return results


def positive_sentences(tokenizer, model, sentences, batch_size=32):
    results = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.Softmax(dim=1)(outputs.logits)
        preds = torch.argmax(probs, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        for j, p in enumerate(preds):
            pos_flags = (p.item() == 1 and max_probs[j].item() >= 0.99)
            results.append(pos_flags)
    return results

'''
Example: Neutral persona filtering. You can adjust the confidence criteria for various polarity levels.

def neutral_sentence(tokenizer, model, sentences, batch_size=32):
    # 결과를 담을 리스트
    results = []
    
    # batch_size 단위로 나누어서 처리
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        # 여러 문장을 한 번에 토크나이징 (padding, truncation 가능)
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.Softmax(dim=1)(logits)
        preds = torch.argmax(probs, dim=1)

        max_probs, _ = torch.max(probs, dim=1) 

        for j, p in enumerate(preds):
            neutral_flags = (max_probs[j].item() < 0.99)
            results.append(neutral_flags)
    
    return results
'''

def NLI(model, description, sample):
    for attribute in description:
        scores = model.predict([(attribute, sample)])
        label_mapping = ['contradiction', 'entailment', 'neutral']
        labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
        if labels[0] == 'contradiction':
            return False            
    return True
    
    
from parlai.core.params import ParlaiParser
from parlai.core.worlds import create_task
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from tqdm import tqdm
import random, json
from sentence_transformers import CrossEncoder
random.seed(42)


def main():
    args = get_args()

    # Load ConvAI2 personas
    parser = ParlaiParser()
    opt = parser.parse_args(args=['-t', 'convai2', '-dt', 'train'])
    agent = RepeatLabelAgent(opt)
    world = create_task(opt, agent)

    persona_descriptions = []
    for _ in tqdm(range(world.num_examples())):
        world.parley()
        msg = world.get_acts()[0]
        person_lines = msg['text'].split("\n")
        for att in person_lines:
            if att.startswith('your persona: '):
                persona_descriptions.append(att[len('your persona: '):])
        world.reset()
    unique_personas = list(set(persona_descriptions))

    # Sentiment filtering
    tokenizer = AutoTokenizer.from_pretrained(args.sent_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.sent_model).to("cuda")
    
    negative_flags = negative_sentences(tokenizer, model, unique_personas, batch_size=args.batch_size)
    negative_personas = [s for s, neg in zip(unique_personas, negative_flags) if neg]

    positive_flags = positive_sentences(tokenizer, model, unique_personas, batch_size=args.batch_size)
    positive_personas = [s for s, pos in zip(unique_personas, positive_flags) if pos]

    # Consistency check via NLI
    
    nli = CrossEncoder(args.nli_model, device="cuda")

    # Build profiles
    positive_user_profile = []
    negative_user_profile = []

    pbar = tqdm(total=args.N)

    while len(positive_user_profile) < args.N:
        positive_persona_description = []
        negative_persona_description = []
        pos = random.choice(positive_personas)
        neg = random.choice(negative_personas)
        positive_persona_description.append(pos)
        negative_persona_description.append(neg)
        while len(positive_persona_description) < args.K:
            pos = random.choice(positive_personas)
            neg = random.choice(negative_personas)
            if (pos not in positive_persona_description and
                neg not in negative_persona_description and
                NLI(nli, positive_persona_description, pos) and
                NLI(nli, negative_persona_description, neg)):
                    positive_persona_description.append(pos)
                    negative_persona_description.append(neg)
        if (positive_persona_description not in positive_user_profile) and (negative_persona_description not in negative_user_profile):
            positive_user_profile.append(positive_persona_description)
            negative_user_profile.append(negative_persona_description)
            pbar.update(1)

    with open(f'positive_user_profiles_{args.K}.json', 'w', encoding='utf-8') as f:
        json.dump(positive_user_profile, f, ensure_ascii=False, indent=4)
    
    with open(f'negative_user_profiles_{args.K}.json', 'w', encoding='utf-8') as f:
        json.dump(negative_user_profile, f, ensure_ascii=False, indent=4)    


    # Mixed profile construction
    mixed_user_profile = []
    binary_set = [positive_personas, negative_personas]
    pbar = tqdm(total=args.N)

    while len(mixed_user_profile) < args.N:
        mixed_persona_description = []
        sampled = []
        random_set = random.randint(0, 1)
        mix = random.choice(binary_set[random_set])
        mixed_persona_description.append(mix)
        while len(mixed_persona_description) < args.K:
            random_set = random.randint(0, 1)
            mix = random.choice(binary_set[random_set])
            if (mix not in mixed_persona_description and 
                NLI(nli, mixed_persona_description, mix)):
                    mixed_persona_description.append(mix)
        if mixed_persona_description not in mixed_user_profile:
            mixed_user_profile.append(mixed_persona_description)
            pbar.update(1)
            
    with open(f'mixed_user_profiles_{args.K}.json', 'w', encoding='utf-8') as f:
        json.dump(mixed_user_profile, f, ensure_ascii=False, indent=4)  

if __name__ == '__main__':
    main()