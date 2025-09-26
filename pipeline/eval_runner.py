from typing import List, Dict, Any
import torch
from tqdm import tqdm
from pipeline.utils import get_device

def generate_text_for_batch(model, tokenizer, prompts: List[str], device, max_new_tokens=128):
    # Works for both causal and seq2seq models (simple approach)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    texts = tokenizer.batch_decode(out, skip_special_tokens=True)
    return texts

def run_eval(model, tokenizer, data: List[Dict], prompt_template: str,
             metric, device, batch_size:int=8, max_new_tokens:int=128):
    predictions = []
    references = []
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        prompts = []
        for item in batch:
            prompt = prompt_template.format(input=item.get("input", ""), target=item.get("target", ""))
            prompts.append(prompt)
        preds = generate_text_for_batch(model, tokenizer, prompts, device, max_new_tokens=max_new_tokens)
        # post-process: for many models the tokenizer.decode(out) returns entire input+output,
        # so we may want to strip prompt prefix:
        results = []
        for p_text, prompt in zip(preds, prompts):
            # attempt to remove prompt prefix
            if p_text.startswith(prompt):
                out_text = p_text[len(prompt):].strip()
            else:
                out_text = p_text.strip()
            results.append(out_text)
        predictions.extend(results)
        references.extend([item.get("target","") for item in batch])

    scores = metric.compute(predictions, references)
    return scores, predictions, references
