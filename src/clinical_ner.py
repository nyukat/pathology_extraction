import torch 
import pandas as pd 
import numpy as np
import re 
import argparse
import os
import pickle
import nltk

from nltk import sent_tokenize
from nltk.tokenize import TreebankWordTokenizer, RegexpTokenizer
from typing import List, Tuple, Union, Dict
from transformers import BertTokenizerFast, BertForTokenClassification, logging
from torch.utils.data import Dataset, DataLoader
from src.dataset import DatasetNER
from src.utils import clean_text, process_punctuation, get_entity_spans

nltk.download('punkt')


def get_ids_to_labels():
    ids_to_labels = {
        0: 'O', 
        1: 'I-cancer_subtype',
        2: 'I-cancer_grade', 
        3: 'I-position',
    }
    return ids_to_labels

def load_model(
    path: str, 
    num_labels: int = 4, 
    pretrained_weights: str = 'emilyalsentzer/Bio_ClinicalBERT', 
    output_hidden_states: bool = False
) -> BertForTokenClassification:
    """
    load BERT token classification model from path
    """
    model = BertForTokenClassification.from_pretrained(
        pretrained_weights,
        num_labels=num_labels, 
        output_hidden_states=output_hidden_states,
        state_dict=torch.load(path)['model']
    )
    return model


def generate_logits(
    model: BertForTokenClassification, 
    batch: Tuple, 
    chunk_size: int = 60,
    overlap: int  = 20, 
    device: Union[str, torch.device] =None
) -> torch.Tensor:
    """
    generate a logit prediction for each word from a batch produced by dataset object
    If the input text exceeds the 128 token limit for our BERT tokenizer, 
        create a rolling window of chunk size 'chunk_size' 
        and generate predictions for each rolling window
    """
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
        
    stride = chunk_size - overlap 
    output_logit = None
    batch_offsets = None

    model = model.to(device)
    
    outputs = model(
        batch['input_ids'].to(device), 
        attention_mask=batch['attention_mask'].to(device)
    )
    for logit, offsets in zip(
        outputs.logits, 
        batch['offset_mapping']
    ):
        # only generate a prediction from non-masked words
        pred_targets = (offsets[:,0] == 0) * (offsets[:,1] != 0)
        logit_predictions = logit[pred_targets,:]

        if output_logit is not None:
            # calculate the average logit over the overlapped sections 
            actual_overlap = min(overlap, logit_predictions.shape[0])
            output_logit[-actual_overlap:] += logit_predictions[:actual_overlap]
            output_logit[-actual_overlap:] /= 2 
            # combine new and previous logit 
            output_logit = torch.vstack([
                output_logit, 
                logit_predictions[actual_overlap:]])
        else:
            output_logit = logit_predictions

    return output_logit 


def extract_entity_from_text(
    report_text: str,
    weights_path: str ='models/model.all_augmentations.pt', 
    pretrained_weights: str ='emilyalsentzer/Bio_ClinicalBERT',
    device: Union[str, torch.device] = None, 
    chunk_size: int = 60,
    ids_to_labels: Dict[int, str] = None,
    overlap: int = 10, 
    max_len: int = 128,
    n_labels: int = 4,
) -> List[Dict]:
    """
    load model from torch weights file 
    and use model to predict enity tags from string input
    """ 
    # clean and convert report text to a list of words to spans
    word_tokenizer = TreebankWordTokenizer()
    # BERT model and tokenizer
    model = load_model(
        weights_path,
        num_labels=n_labels, 
        pretrained_weights=pretrained_weights
    )
    bert_tokenizer = BertTokenizerFast.from_pretrained(pretrained_weights)
    entity_list = ner_inference(
        report_text,
        model, 
        ids_to_labels=ids_to_labels,
        bert_tokenizer=bert_tokenizer, 
        word_tokenizer=word_tokenizer,
        chunk_size=chunk_size,
        overlap=overlap, 
        max_len=max_len,
        device=device,
    )
    return entity_list


def extract_entity_from_report(filename: str, **kwargs) -> List[Dict]: 
    """
    load model from torch weights file and 
    collect list of clinically relevant entities from .txt file
    """
    with open(filename, 'rb') as f:
        report_text = f.read().decode()
    return extract_entity_from_text(report_text, **kwargs)
    

def ner_inference(
    report_text: str,
    model: BertForTokenClassification, 
    ids_to_labels: Dict[int, str]=None,
    bert_tokenizer: BertTokenizerFast = None, 
    word_tokenizer: RegexpTokenizer = None,
    device: Union[str, torch.device] = None, 
    chunk_size: int = 60,
    overlap: int = 10, 
    max_len: int = 128,
):
    """
    1. report text is divides it into individual sentences. 
    2. each sentence is broken down into individual words using tokenization. 
    3. A BERT model is then used to classify each of these word tokens, 
        and the predicted entity type is assigned to each token. 
    4. Token-level predictions are combined to generate continuous entity 
        predictions for the input text.
    """
    if ids_to_labels is None:
        # map BERT classes to class names
        ids_to_labels = get_ids_to_labels()
        
    sentences = [
        sent for sent in sent_tokenize(
            re.sub(r'\s+', ' ', report_text).strip().lower()
        )
    ]
    span_lists = [list(word_tokenizer.span_tokenize(sent)) for sent in sentences] 
    word_lists = [
        [sent[start:end] for start,end in list(spans)]
        for sent, spans in zip(sentences, span_lists)
    ]
    input_dataset = DatasetNER(
        word_lists, 
        tokenizer=bert_tokenizer, 
        chunk_size=chunk_size,
        overlap=overlap, 
        max_len=max_len,
    )
    entity_list = []
    for spans, sent, tokens, batch in zip(
        span_lists, sentences, word_lists, input_dataset
    ):
        # generate word level labels using NER model 
        output = generate_logits(
            model, batch, 
            chunk_size=chunk_size, 
            overlap=overlap,
            device=device
        )
        raw_labels = output.argmax(axis=-1).detach().tolist()
        labels = process_punctuation(tokens, raw_labels)
        
        # combine word level labels to retrieve texts 
        report_entities = [] 
        entity_spans = get_entity_spans(spans, labels)
        for entity in entity_spans: 
            start,end = entity['span']
            entity['label'] = ids_to_labels[entity['label']]
            entity['text'] = sent[start:end]
            report_entities.append(entity)
            
        entity_list.append({'sentence': sent, 'entities': report_entities})

    return entity_list
    
    
def main():
    """
    read list of report txt files from input directory
    generate a pickle file for each input report
    and save outputs in output directory 
    """
    parser = argparse.ArgumentParser(
        description='Extract key clinical entities from input directory of reports'
    )
    parser.add_argument('--input-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--model-file', default='/models/model.baseline.pt', type=str)
    parser.add_argument('--pretrained-weights', default='emilyalsentzer/Bio_ClinicalBERT', type=str)
    parser.add_argument('--device', default=None, type=str)
    args = parser.parse_args()

    ids_to_labels = {
        0: 'none_entity', 
        1: 'cancer_subtype',
        2: 'cancer_grade', 
        3: 'position',
    }
    print(f'cuda is available: {torch.cuda.is_available()}')
    # initialiize model, word tokenizer and bert tokenizer 
    model = load_model(
        args.model_file,
        num_labels=len(ids_to_labels), 
        pretrained_weights=args.pretrained_weights
    )
    word_tokenizer = TreebankWordTokenizer()
    bert_tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_weights)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    # iterate over available reports
    for file in os.listdir(args.input_dir):
        file_path = os.path.join(args.input_dir, file)
        print(f'file read: {file_path}')
        basename, extension = os.path.splitext(file)

        # read text file as string and use NER to extract entities
        if extension == '.txt':
            with open(file_path, 'rb') as f:
                report_text = f.read().decode()

            entity_list = ner_inference(
                report_text, 
                model=model,
                bert_tokenizer=bert_tokenizer, 
                word_tokenizer=word_tokenizer,
                device=args.device, 
            )
            output_file = os.path.join(
                args.output_dir, f'{basename}.entities.pickle'
            )
            with open(output_file, 'wb') as f:
                pickle.dump(entity_list, f)
            print(f'output file: {output_file}')
    

if __name__ == "__main__":
    
    main()
    
    
