from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification,DistilBertConfig
import torch
from transformers import logging
logging.set_verbosity_error()

def prep_transformer():
    #configuration = DistilBertConfig(max_position_embeddings=2048)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    return model,tokenizer


    