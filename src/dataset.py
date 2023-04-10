import torch 
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
from collections import defaultdict
from typing import List, Tuple, Union, Dict
from src.utils import clean_text


class DatasetNER(Dataset):
    """
    breaks the input text into chunks of length chunk_size with overlap 
    the average logit probability is used for to combine predictions 
        for text within the overlap 
    """
    def __init__(
        self, 
        data: List[List[str]], 
        tokenizer: BertTokenizerFast = None, 
        max_len: int = 128,            
        chunk_size: int = 59,
        overlap: int = 10,
        process: bool = False,
    ):
        self.len = len(data)
        self.data = data      
        self.max_len = max_len
        self.overlap=overlap 
        self.chunk_size=chunk_size
        self.tokenizer = tokenizer
        self.process = process
        assert max_len > self.chunk_size * 2, 'chunk sizd too large' 
        
    def __getitem__(self, index: int):

        sentence = self.data[index]
        sentence = [word.lower() for word in sentence]
        if self.process:
            sentence = clean_text(sentence)
            
        items = defaultdict(lambda:[])
        stride = self.chunk_size - self.overlap 
        for start in range(0,len(sentence), stride):
            
            encoding = self.tokenizer(
                sentence[start: start + self.chunk_size],
                is_split_into_words=True,
                return_offsets_mapping=True, 
                padding='max_length', 
                truncation=True, 
                max_length=self.max_len
            )
            for key,val in encoding.items():
                items[key].append(val)
        items = {key: torch.as_tensor(val) for key, val in items.items()}
        return items

    def __len__(self):
        return self.len
