import pandas as pd 
import numpy as np
import argparse
import pickle
import torch 
import logging
from seqeval.metrics import classification_report
from typing import List, Tuple, Union, Dict
from transformers import BertTokenizerFast, BertForTokenClassification
from src.dataset import DatasetNER
from src.utils import process_punctuation, get_entity_spans
from src.clinical_ner import generate_logits, load_model, get_ids_to_labels


def ner_inference_tokenized_text(
    reports: List[List[str]],
    model: BertForTokenClassification, 
    ids_to_labels: Dict[int, str]=None,
    bert_tokenizer: BertTokenizerFast = None, 
    device: Union[str, torch.device] = None, 
    chunk_size: int = 60,
    overlap: int = 10, 
    max_len: int = 128,
    process: bool = False, 
) -> List[List[str]]:
    
    input_dataset = DatasetNER(
        reports, 
        tokenizer=bert_tokenizer, 
        chunk_size=chunk_size,
        overlap=overlap, 
        max_len=max_len,
        process=process,
    )
    output_labels = [] 
    for report, batch in zip(reports, input_dataset):
        # generate word level labels using NER model 
        output = generate_logits(
            model, 
            batch, 
            chunk_size=chunk_size, 
            overlap=overlap,
            device=device
        )
        raw_labels = output.argmax(axis=-1).detach().tolist()
        labels = process_punctuation(report, raw_labels)
        output_labels.append([ids_to_labels[x] for x in labels])
        
    return output_labels 

def main():
    """
    read list of report txt files from input directory
    generate a pickle file for each input report
    and save outputs in output directory 
    """
    parser = argparse.ArgumentParser(
        description='Extract key clinical entities from input directory of reports'
    )
    parser.add_argument('--input-file', required=True, type=str)
    parser.add_argument('--output-file', default=None, type=str)
    parser.add_argument('--model-file', 
                        default='/models/model.baseline.pt', type=str)
    parser.add_argument('--pretrained-weights', 
                        default='emilyalsentzer/Bio_ClinicalBERT', type=str)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--prediction-column', default='predicted_labels', type=str)
    parser.add_argument('--label-column', default='cleaned_labels', type=str)
    parser.add_argument('--report-column', default='reports')
    parser.add_argument('--evaluate-outputs', default=True, type=bool)
    parser.add_argument('--process', default=True, type=bool)
    args = parser.parse_args()
    
    with open(args.input_file, 'rb') as f:
        report_labels = pickle.load(f)
        
    ids_to_labels = get_ids_to_labels()
    
    model = load_model(
        args.model_file,
        num_labels=len(ids_to_labels), 
        pretrained_weights=args.pretrained_weights
    )
    bert_tokenizer = BertTokenizerFast.from_pretrained(args.pretrained_weights)
    
    outputs = ner_inference_tokenized_text(
        report_labels[args.report_column],
        model, 
        bert_tokenizer=bert_tokenizer, 
        ids_to_labels=ids_to_labels,
        device=args.device, 
        process=args.process,
    )
    
    report_labels = report_labels.assign(**{str(args.prediction_column):outputs})
    if args.evaluate_outputs:
        
        # evaluate NER performanced
        performance = classification_report(
            report_labels[args.label_column], 
            report_labels[args.prediction_column], 
            output_dict=True
        )  
        logging.basicConfig(level=logging.INFO)
        logging.info('NER metrics')
        logging.info(performance)
        
    if args.output_file is not None:
        with open(args.output_file, 'wb') as f:
            pickle.dump(report_labels, f)

    
if __name__ == "__main__":
    
    main() 