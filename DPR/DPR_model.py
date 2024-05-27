import transformers
from transformers import BertModel
import torch
import os
from copy import deepcopy

class KobertBiEncoder(torch.nn.Module):
    def __init__(self):
        super(KobertBiEncoder, self).__init__()
        self.passage_encoder = BertModel.from_pretrained("klue/bert-base")
        self.query_encoder = BertModel.from_pretrained("klue/bert-base")

    def forward(
        self, x: torch.LongTensor, attn_mask: torch.LongTensor, type: str = "passage"
    ) -> torch.FloatTensor:
        """passage 또는 query를 bert로 encoding합니다."""
        assert type in (
            "passage",
            "query",
        ), "type should be either 'passage' or 'query'"
        if type == "passage":
            return self.passage_encoder(
                input_ids=x, attention_mask=attn_mask
            ).pooler_output
        else:
            return self.query_encoder(
                input_ids=x, attention_mask=attn_mask
            ).pooler_output
