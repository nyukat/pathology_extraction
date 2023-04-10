from typing import List, Tuple, Union, Dict


PUNCTUATION = [',', '-', ':', '', '"', '\'', ';', '.']


def get_entities(labels: List[str]):
    """
    get_start and end indices 
    """
    entities = []; curr = None; prev = 0
    for i, label in enumerate(labels + [0]):
        if label != prev:
            new = None
            if label != 0:
                new = [label, i, None]
            if curr:
                curr[2] = i-1
                entities.append(curr)
            curr = new
        prev = label
    return entities


def clean_text(report: List[str]):
    cleaned_report = []
    for word in report:
        modified_word = word 
        if len(word) > 1:
            if word[0] in ['·']:
                modified_word = modified_word[1:]
            if word[-1] in ['.'] and len(word) > 4:
                 modified_word = modified_word[:-1]
        cleaned_report.append(modified_word)
    return cleaned_report 


def process_punctuation(
    report: str, 
    tags: List[str], 
    punctuation: List[str]=None
):
    """clean NER labels for punctuations"""
    if punctuation is None:
        punctuation = PUNCTUATION
    new_tags = [x for x in tags]
    for i in range(1, len(tags)-1):
        if report[i] in punctuation:
            if tags[i-1] == tags[i+1]:
                new_tags[i] = tags[i+1]
            else: # an entity should never start or end with punctuation
                new_tags[i] = 0 # non-entity label 
    
    entities = get_entities(tags)
    for label, start, end in entities:
        if label == 2 and start == end: 
            if report[start] in ['grade', '-grade']:
                new_tags[start] = 0
    return new_tags


def get_entity_spans(
    spans: List[Tuple[int,int]], 
    labels: List[str]
) -> List[Dict]:
    """
    Given a set of word token labels and the span of each word 
    retrieve the span of each entity in the text 
    """
    entity_start = None
    entity_end = None
    prev_label = None
    entities = [] 
    for i, ((span_start, span_end),label) in enumerate(zip(spans, labels)):
        # close current entity:
        if label != prev_label: 
            if prev_label not in [0, None]: # 0 is for non-entities
                entity = {
                    'label': prev_label, 
                    'span': (entity_start, entity_end)
                }
                entities.append(entity)
            entity_start = span_start
            prev_label = label
        entity_end = span_end
        
    if prev_label != 0:
        entity = {'label': prev_label, 'span': (entity_start, entity_end)}
        entities.append(entity)
    return entities
    
    