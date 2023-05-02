import torch
from torchtext.data import get_tokenizer
from torch.utils.data import TensorDataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.functional import to_tensor, truncate
from torch.utils.data import TensorDataset

tokenize = get_tokenizer("basic_english")

def iter_review_tokens(df):
    reviews_iter = iter(df['review'].apply(tokenize))
    return reviews_iter
    

def get_vocabulary(df, max_tokens):
    special_tokens = ['<pad>', '<unk>', '<start>']
    voc = build_vocab_from_iterator(iter_review_tokens(df), specials= special_tokens, max_tokens= max_tokens)
    return voc
    
def get_review_dataset(df, voc, max_length):
    # Truncate all token sequences
    reviews_truncated = [truncate(token, max_length) for token in iter_review_tokens(df)]
    
    # Encode tokens
    encoded_reviews = list([voc.lookup_indices(x) for x in reviews_truncated])
    
    #Pad tokens
    padded_reviews = to_tensor(encoded_reviews, padding_value=voc['<pad>'])
    
    #Encode labels as 0's or 1's and change to a tensor
    labels = [0 if label == 'negative' else 1 for label in df['label'].values]
    labels = to_tensor(labels)
    
    # Create Dataset
    dataset = TensorDataset(padded_reviews, labels)
    
    return dataset

