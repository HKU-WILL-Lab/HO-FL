import torch

def collate(tokenized, indices, pad_id, device):
    seqs = [tokenized['input_ids'][i] for i in indices]
    width = max(map(len, seqs))
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for j, seq in enumerate(seqs):
        ids[j, -len(seq):] = torch.tensor(seq)
        mask[j, -len(seq):] = 1
    labels = torch.tensor([tokenized['labels'][i] for i in indices], dtype=torch.long)
    return {'input_ids': ids.to(device), 'attention_mask': mask.to(device), 'labels': labels.to(device)}

class ClientStream:

    def __init__(self, client, batch_size):
        self.indices = client['indices']
        self.batch_size = batch_size
        self.generator = torch.Generator().manual_seed(client['data_seed'])
        self.order = []
        self.cursor = 0
        self.batches_seen = 0

    def next(self):
        if self.cursor + self.batch_size > len(self.order):
            self.order = torch.randperm(len(self.indices), generator=self.generator).tolist()
            self.cursor = 0
        batch = [self.indices[j] for j in self.order[self.cursor:self.cursor + self.batch_size]]
        self.cursor += self.batch_size
        self.batches_seen += 1
        assert len(batch) == self.batch_size
        return batch

def collate_answers(data, indices, pad_id, device):
    prompts = [data['input_ids'][i] for i in indices]
    answers = [data['answer_ids'][i] for i in indices]
    width = max((len(p) + len(a) for p, a in zip(prompts, answers)))
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    targets = torch.full_like(ids, -100)
    for row, (prompt, answer) in enumerate(zip(prompts, answers)):
        sequence = prompt + answer
        offset = width - len(sequence)
        ids[row, offset:] = torch.tensor(sequence)
        attention[row, offset:] = 1
        start = offset + len(prompt) - 1
        targets[row, start:start + len(answer)] = torch.tensor(answer)
    prediction_mask = targets != -100
    assert not prediction_mask[:, -1].any()
    return {'input_ids': ids.to(device), 'attention_mask': attention.to(device), 'prediction_mask': prediction_mask.to(device), 'labels': targets[prediction_mask].to(device)}

def collate_prompts(data, indices, pad_id, device):
    prompts = [data['input_ids'][i] for i in indices]
    width = max(map(len, prompts))
    ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    for row, prompt in enumerate(prompts):
        ids[row, -len(prompt):] = torch.tensor(prompt)
        attention[row, -len(prompt):] = 1
    return {'input_ids': ids.to(device), 'attention_mask': attention.to(device)}
