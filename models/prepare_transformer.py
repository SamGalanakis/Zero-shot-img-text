from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification,DistilBertConfig
import torch


def prep_transformer():
    configuration = DistilBertConfig(max_position_embeddings=2048)
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained(configuration)\
    
    return model,tokenizer


    