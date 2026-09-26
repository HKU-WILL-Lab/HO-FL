import numpy as np
from .io import write_json

PROMPT_TOKENS = 512
ANSWER_TOKENS = 64
QUESTION_TOKENS = 128
EVAL_SUBSET_SIZE = 1024
EVAL_SUBSET_SEED = 190517
TOKENIZATION_VERSION = 1

def shuffled_options(row, rng):
    options = [row['correct_answer'], row['distractor1'], row['distractor2'], row['distractor3']]
    order = rng.permutation(4).tolist()
    return ([options[i] for i in order], order.index(0))

def sciq_tokens(row, tokenizer, rng):
    options, label = shuffled_options(row, rng)
    suffix = ''.join((f'\n{letter}: {option}' for letter, option in zip('ABCD', options))) + '\nAnswer:'
    tail = tokenizer.encode(suffix, add_special_tokens=False)
    head = tokenizer.encode('Question: ' + row['question'], add_special_tokens=False)
    budget = 256 - len(tail) - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 0:
        raise ValueError('SciQ options exceed token budget; never silently truncate options')
    ids = tokenizer.build_inputs_with_special_tokens(head[:budget] + tail)
    assert len(ids) <= 256
    return (ids, label)

def squad_tokens(dataset, tokenizer, asset, dataset_asset):
    prefix = tokenizer.encode('Context:', add_special_tokens=False)
    question_marker = tokenizer.encode('\nQuestion:', add_special_tokens=False)
    answer_marker = tokenizer.encode('\nAnswer:', add_special_tokens=False)
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    result = {'metadata': {'version': TOKENIZATION_VERSION, 'model_revision': asset['revision'], 'dataset_revision': dataset_asset['revision'], 'prompt_tokens': PROMPT_TOKENS, 'answer_tokens_including_eos': ANSWER_TOKENS, 'prompt_template': 'Context: <context>\\nQuestion: <question>\\nAnswer:', 'context_truncation': 'right; question and answer delimiter always retained; no gold-conditioned window selection', 'loss_reduction': 'mean over answer and EOS tokens only', 'eos_token_id': tokenizer.eos_token_id}}
    for split in ['train', 'validation']:
        records = {'input_ids': [], 'answer_ids': [], 'ids': [], 'references': [], 'context_truncated': [], 'reference_visible': [], 'answer_truncated': [], 'question_truncated': []}
        for start in range(0, len(dataset[split]), 512):
            batch = dataset[split][start:start + 512]
            contexts = tokenizer([' ' + c for c in batch['context']], add_special_tokens=False, return_offsets_mapping=True)
            questions = tokenizer([' ' + q for q in batch['question']], add_special_tokens=False)['input_ids']
            suffixes = [question_marker + q[:QUESTION_TOKENS] + answer_marker for q in questions]
            first_answers = tokenizer([' ' + a['text'][0] for a in batch['answers']], add_special_tokens=False)['input_ids']
            for i, suffix in enumerate(suffixes):
                budget = PROMPT_TOKENS - len(prefix) - len(suffix) - special_count
                assert budget > 0, batch['id'][i]
                context_ids = contexts['input_ids'][i]
                kept = min(budget, len(context_ids))
                prompt = tokenizer.build_inputs_with_special_tokens(prefix + context_ids[:kept] + suffix)
                assert 0 < len(prompt) <= PROMPT_TOKENS
                answer = first_answers[i][:ANSWER_TOKENS - 1] + [tokenizer.eos_token_id]
                end_char = contexts['offset_mapping'][i][kept - 1][1]
                gold = batch['answers'][i]
                visible = any((pos + 1 + len(text) <= end_char for pos, text in zip(gold['answer_start'], gold['text'])))
                records['input_ids'].append(prompt)
                records['answer_ids'].append(answer)
                records['ids'].append(batch['id'][i])
                records['references'].append(gold['text'])
                records['context_truncated'].append(len(context_ids) > budget)
                records['reference_visible'].append(visible)
                records['answer_truncated'].append(len(first_answers[i]) >= ANSWER_TOKENS)
                records['question_truncated'].append(len(questions[i]) > QUESTION_TOKENS)
        assert len(set(records['ids'])) == len(records['ids'])
        result[split] = records
        result['metadata'][split] = {'examples': len(records['ids']), 'context_truncated_fraction': float(np.mean(records['context_truncated'])), 'no_reference_fully_visible_fraction': float(1 - np.mean(records['reference_visible'])), 'answer_target_truncated_fraction': float(np.mean(records['answer_truncated'])), 'mean_prompt_tokens': float(np.mean(list(map(len, records['input_ids'])))), 'mean_answer_tokens': float(np.mean(list(map(len, records['answer_ids']))))}
        result['metadata'][split]['question_truncated_fraction'] = float(np.mean(records['question_truncated']))
    result['metadata']['question_token_limit'] = QUESTION_TOKENS
    rng = np.random.default_rng(EVAL_SUBSET_SEED)
    subset = sorted(rng.choice(len(result['validation']['ids']), EVAL_SUBSET_SIZE, replace=False).tolist())
    result['evaluation_indices'] = subset
    result['metadata']['evaluation_subset_seed'] = EVAL_SUBSET_SEED
    result['metadata']['evaluation_subset_ids'] = [result['validation']['ids'][i] for i in subset]
    return result
