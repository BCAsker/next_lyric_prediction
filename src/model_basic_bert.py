from torch import nn
from transformers import BertForNextSentencePrediction

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from transformers import BertTokenizer, BertModel


class BasicBertPredictor(nn.Module):
    def __init__(self, checkpoint="bert-base-cased"):
        super(BasicBertPredictor, self).__init__()
        self.bert_model = BertForNextSentencePrediction.from_pretrained(checkpoint)

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).logits
