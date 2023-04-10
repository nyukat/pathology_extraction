 #!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH

python3 src/clinical_ner.py \
    --input-dir 'data' \
    --output-dir 'example_outputs' \
    --model-file 'models/model.all_augmentations.pt' \
    --pretrained-weights 'emilyalsentzer/Bio_ClinicalBERT' \
    --device 'cpu'