from transformers import LongformerModel,DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification,DistilBertConfig
import torch
import transformers
from transformers import logging
from transformers.utils.dummy_tokenizers_objects import LongformerTokenizerFast
from sentence_transformers import SentenceTransformer, util
logging.set_verbosity_error()

def prep_transformer():
    #configuration = DistilBertConfig(max_position_embeddings=2048)
    # tokenizer = transformers.LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
    # model = transformers.LongformerModel.from_pretrained('allenai/longformer-base-4096')
    model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')
    return model


    