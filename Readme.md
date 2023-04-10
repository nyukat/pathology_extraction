# Improving Information Extraction from Pathology Reports using Named Entity Recognition

## Introduction

This is an implementation of the *model_name* model as described in [our paper](). The architecture of the proposed model is shown below.

We present a transformer-based named entity recognition (NER) system that effectively extracts key elements of diagnosis from pathology reports. As the annotation of data for NER can be time-consuming, we employed data augmentation techniques to generate additional training data from the available pathology reports to finetune a [Clinical BERT model](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) to identify cancer subtype, cancer grade and position from NYU breast cancer pathology reports.

We evaluated and trained our model on a dataset of 1438 annotated reports and achieved a high entity F1-score of 0.916 on the test set. Notably, this performance surpasses that of a strong BERT baseline, which achieved an F1-score of 0.843 on the same dataset.

These results highlight the effectiveness of our approach in capturing important contextual information from pathology reports that may be challenging for heuristic methods to capture. Our tool can assist doctors in quickly and accurately extracting key information from pathology reports, potentially improving patient outcomes and reducing diagnostic errors.

<p align="center">
  <img width="700" height="624" src="https://github.com/nyukat/pathology_extraction/blob/main/figures/figure1.png">
</p>

## Prerequisites

* Python (3.9.16)
* PyTorch (1.8.0)
* torchvision (0.9.0)
* NumPy (1.23.5)
* transformers (4.24.0)
* imageio (2.4.1)
* pandas (2.0.0)
* tqdm (4.65.0)
* nltk (3.7)
* seqeval (1.2.2)

## License

This repository is licensed under the terms of the GNU AGPLv3 license.

## Data

The examples in `data` used for testing contained 7 diagnosis examples from NYU pathology reports. Notably, the tool was able to successfully navigate 5 of these cases that presented challenges for our existing rule-based approach. Specifically, the tool was able to accurately handle 3 cases with misspellings, a difficulty that strict rule-based systems often struggle with. Furthermore, the tool demonstrated its contextual understanding by accurately interpreting 2 cases that included negation, a task that requires a high level of precision and attention to detail. Overall, our tool has proven to be a valuable asset for doctors looking for a reliable and efficient way to navigate complex diagnosis scenarios.

. Here's an example of one of the diagnosis: 

```
1. SENTINEL LYMPH NODE, LEFT AXILLA, BIOPSY:
- THREE LYMPH NODES, NEGATIVE FOR CARCINOMA (0/3)
Comment: The immunostain for the cytokeratin AE1/AE3 is negative.
2. SENTINEL LYMPH NODE #2, LEFT AXILLA, BIOPSY:
- ONE LYMPH NODE, NEGATIVE FOR CARCINOMA (0/1).
Comment: The immunostain for the cytokeratin AE1/AE3 is negative.
3. NON-SENTINEL LYMPH NODE, LEFT AXILLA 'PALPABLE', BIOPSY:
- TWO LYMPH NODES, NEGATIVE FOR CARCINOMA (0/2)
4. BREAST, LEFT, TOTAL MASTECTOMY:
- DUCTAL CARCINOMA IN SITU, INTERMEDIATE NUCLEAR GRADE
The DCIS is seen as a single focus measuring 0.1 cm, and is
morphologically similar to the prior DCIS (see xxxx).
Negative margins
- FIBROCYSTIC CHANGE, PROLIFERATIVE TYPE, ASSOCIATED WITH
MICROCALCIFICATIONS
- DERMAL SCAR AND CHANGES OF PRIOR PROCEDURE
- NIPPLE WITH NO PATHOLOGIC ALTERATIONS
5. SKIN, LEFT BREAST, EXCISION:
- SKIN AND SUBCUTANEOUS TISSUE WITHOUT PATHOLOGIC ALTERATIONS
```

## How to run the code

You need to first install conda in your environment. **Before running the code, please run `pip install -r requirements.txt` first.** Once you have installed all the dependencies, `run.sh` will automatically run the entire pipeline and save the prediction results in csv. Note that you need to first cd to the project directory and then execute `. ./run.sh`. When running the individual Python scripts, please include the path to this repository in your `PYTHONPATH`. We recommend running the code with a GPU. To run the code with CPU only, please change `device` in run.sh to 'cpu'. 

Alternatively, you can directly call functions from `src.clinical_ner` to parse the text file: 

```python 

# to parse a txt file 
outputs = ner.extract_entity_from_report(
    'data/example_diagnosis.negation_1.txt',
    weights_path ='models/model.all_augmentations.pt'
)

# to directly parse input string 
example_text =  "a. breast, left 11:00 o'clock 5cm from the nipple - ductal carcinoma in situ"

outputs = ner.extract_entity_from_text(
    example_text,
    weights_path ='models/model.all_augmentations.pt'
)
```

## What do the outputs mean? 

We provided the example outputs to each example report in the `example_outputs` folder. You can use `pickle` package to open the outputs from our pipeline. 

```python 
import pickle

with open(entities_output_filename, 'rb') as f:
    example_output = pickle.load(f)
```

In our pipeline, we first utilize `nltk` to divide the report text into sentences. After that, we implement our BERT model to make predictions regarding the entities present in the input text. Our output includes the start and end indexes of the corresponding entity, along with the entity class and the text encompassed by the entity. In the following example, our model extracted two entities from the input text, which only contains one sentence. The first entity is the position of the finding, `11:00 o'clock 5cm from the nipple` and along with the span. 

```python
[
  {
    'sentence': "a. breast, left 11:00 o'clock 5cm from the nipple - ductal carcinoma in situ",
    'entities': [
      {
        'label': 'position',
        'span': (16, 49),
        'text': "11:00 o'clock 5cm from the nipple"
      },
      {
        'label': 'cancer_subtype',
        'span': (52, 76),
        'text': 'ductal carcinoma in situ'
      }
    ]
  }
]
```

## Additional information

We cannot make the dataset public, but we will evaluate models from other research
institutions on the test part of the data set upon request. For any further queries regarding data availability, please contact (k.j.geras@nyu.edu).

## Reference

If you found this code useful, please cite our paper:

**"Improving Information Extraction from Pathology Reports using Named Entity Recognition"**

Ken G. Zeng, Tarun Dutt, Jan Witowski, Kranthi Kiran GV, Frank Yeung, Michelle
Kim, Jesi Kim3, Mitchell Pleasure, Christopher Moczulski, L. Julian Lechuga Lopez,
Hao Zhang, Mariam Al Harbi, Farah E. Shamout, Vincent J. Major, Laura Heacock,
Linda Moy, Freya Schnabel, Linda Pak, Yiqiu Shen, and Krzysztof J. Geras


