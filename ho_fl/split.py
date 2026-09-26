import torch
import torch.nn.functional as F
from transformers import GenerationConfig
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

class Partition:

    def __init__(self, causal_lm):
        self.lm = causal_lm
        self.body = causal_lm.model
        self.layers = self.body.layers

    def forward_prefix(self, input_ids, attention_mask, cut):
        assert 0 <= cut <= len(self.layers)
        hidden, context = self.embed(input_ids, attention_mask)
        for layer in self.layers[:cut]:
            hidden = self.block(layer, hidden, context)
        return (hidden, context)

    def forward_suffix(self, hidden, context, cut, last_only=False, token_ids=None):
        for layer in self.layers[cut:]:
            hidden = self.block(layer, hidden, context)
        hidden = self.finish(hidden)
        if last_only:
            hidden = hidden[:, -1:, :]
        if token_ids is None:
            logits = self.lm.lm_head(hidden)
        else:
            weight = self.lm.lm_head.weight[token_ids]
            logits = F.linear(hidden, weight)
        cap = getattr(self.lm.config, 'final_logit_softcapping', None)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def finish(self, hidden):
        return self.body.norm(hidden)

    def forward(self, input_ids, attention_mask, token_ids=None):
        hidden, context = self.forward_prefix(input_ids, attention_mask, 0)
        return self.forward_suffix(hidden, context, 0, last_only=token_ids is not None, token_ids=token_ids)

class QwenPartition(Partition):

    def embed(self, input_ids, attention_mask):
        hidden = self.body.embed_tokens(input_ids)
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        position_ids = cache_position.unsqueeze(0)
        kwargs = dict(config=self.body.config, input_embeds=hidden, attention_mask=attention_mask, cache_position=cache_position, past_key_values=None, position_ids=position_ids)
        masks = {'full_attention': create_causal_mask(**kwargs)}
        if self.body.has_sliding_layers:
            masks['sliding_attention'] = create_sliding_window_causal_mask(**kwargs)
        return (hidden, dict(masks=masks, position_ids=position_ids, cache_position=cache_position, position_embeddings=self.body.rotary_emb(hidden, position_ids)))

    def block(self, layer, hidden, ctx):
        return layer(hidden, attention_mask=ctx['masks'][layer.attention_type], position_ids=ctx['position_ids'], cache_position=ctx['cache_position'], position_embeddings=ctx['position_embeddings'], past_key_values=None, use_cache=False)

class OPTPartition(Partition):

    def __init__(self, causal_lm):
        self.lm = causal_lm
        self.body = causal_lm.model.decoder
        self.layers = self.body.layers

    def embed(self, input_ids, attention_mask):
        hidden = self.body.embed_tokens(input_ids)
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        mask = self.body._update_causal_mask(attention_mask, hidden, cache_position, None, False)
        position_ids = (torch.cumsum(attention_mask, dim=1) * attention_mask - 1).long()
        positions = self.body.embed_positions(attention_mask, 0, position_ids=position_ids)
        if self.body.project_in is not None:
            hidden = self.body.project_in(hidden)
        return (hidden + positions.to(hidden.device), dict(mask=mask, position_ids=position_ids, cache_position=cache_position))

    def block(self, layer, hidden, ctx):
        return layer(hidden, attention_mask=ctx['mask'], position_ids=ctx['position_ids'], cache_position=ctx['cache_position'], layer_head_mask=None, past_key_values=None, use_cache=False, output_attentions=False)[0]

    def finish(self, hidden):
        if self.body.final_layer_norm is not None:
            hidden = self.body.final_layer_norm(hidden)
        if self.body.project_out is not None:
            hidden = self.body.project_out(hidden)
        return hidden

class SmolLM3Partition(QwenPartition):

    def __init__(self, causal_lm):
        assert causal_lm.config.model_type == 'smollm3'
        super().__init__(causal_lm)
        self.lm.generation_config = GenerationConfig()
