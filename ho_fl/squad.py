import torch
import numpy as np
from transformers import GenerationConfig
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1, normalize_answer
from .batches import collate_answers, collate_prompts
from .algorithms import task_loss

ANSWER_TOKENS = 64
GENERATION_BATCH = 32

class AnswerLossWrapper:

    def __init__(self, partition, batch):
        self.partition = partition
        self.prediction_mask = batch['prediction_mask']

    def forward_prefix(self, input_ids, attention_mask, cut):
        return self.partition.forward_prefix(input_ids, attention_mask, cut)

    def forward_suffix(self, hidden, context, cut, last_only=False, token_ids=None):
        assert token_ids is None
        p = self.partition
        for layer in p.layers[cut:]:
            hidden = p.block(layer, hidden, context)
        hidden = p.finish(hidden)
        logits = p.lm.lm_head(hidden[self.prediction_mask])
        cap = getattr(p.lm.config, 'final_logit_softcapping', None)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits[:, None, :]

    def forward(self, input_ids, attention_mask, token_ids=None):
        hidden, context = self.forward_prefix(input_ids, attention_mask, 0)
        return self.forward_suffix(hidden, context, 0, token_ids=token_ids)

def score_v1(prediction, references):
    assert references
    return {'exact_match': float(max((compute_exact(gold, prediction) for gold in references))), 'f1': float(max((compute_f1(gold, prediction) if normalize_answer(gold) else 0.0 for gold in references)))}

def generation_config(tokenizer, use_cache=True):
    return GenerationConfig(max_new_tokens=ANSWER_TOKENS, min_new_tokens=0, do_sample=False, num_beams=1, num_return_sequences=1, use_cache=use_cache, repetition_penalty=1.0, no_repeat_ngram_size=0, length_penalty=1.0, early_stopping=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, bos_token_id=tokenizer.bos_token_id, stop_strings=['\n'])

def aggregate_predictions(records):
    assert records and len({row['id'] for row in records}) == len(records)
    return {'exact_match': 100 * float(np.mean([r['exact_match'] for r in records])), 'f1': 100 * float(np.mean([r['f1'] for r in records])), 'count': len(records)}

@torch.no_grad()
def generate_answers(partition, tokenizer, data, indices, batch_size=GENERATION_BATCH, use_cache=True):
    records = []
    config = generation_config(tokenizer, use_cache)
    device = next(partition.lm.parameters()).device
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        batch = collate_prompts(data, selected, tokenizer.pad_token_id, device)
        generated = partition.lm.generate(**batch, generation_config=config, tokenizer=tokenizer)
        generated = generated[:, batch['input_ids'].shape[1]:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for index, text in zip(selected, texts):
            prediction = text.split('\n', 1)[0].strip()
            records.append({'id': data['ids'][index], 'prediction': prediction, 'raw_generation': text, 'references': data['references'][index], 'reference_visible': data['reference_visible'][index], **score_v1(prediction, data['references'][index])})
        if start == 0 or (start // batch_size + 1) % 20 == 0:
            print(__import__("json").dumps({'event': 'GENERATING', 'completed': len(records), 'total': len(indices)}), flush=True)
    return (aggregate_predictions(records), records)

@torch.no_grad()
def answer_loss(partition, tokenizer, data, indices, batch_size=16):
    total, tokens = (0.0, 0)
    device = next(partition.lm.parameters()).device
    for start in range(0, len(indices), batch_size):
        batch = collate_answers(data, indices[start:start + batch_size], tokenizer.pad_token_id, device)
        loss = task_loss(AnswerLossWrapper(partition, batch), batch, None)
        assert torch.isfinite(loss)
        count = batch['labels'].numel()
        total += loss.item() * count
        tokens += count
    return total / tokens
