# Improving Information Extraction from Pathology Reports using Named Entity Recognition

## Introduction
This is an implementation of the *model_name* model as described in [our paper](). The architecture of the proposed model is shown below.

<p align="center">
  <img width="793" height="729" src="https://github.com/pathology_parsing/figures/">
</p>

In this work, we propose a transformer-based named entity recognition (NER) system to extract key elements of diagnosis from pathology reports. Trained and evaluated on 1438 annotated breast cancer pathology reports, our model achieved a entity F1-score of 0.916 on the test set, surpassing a strong BERT baseline (0.843). The experiment results demonstrate that our model can effectively utilize contextual information that would be challenging for heuristic methods to capture.

## Prerequisites

* Python (3.6)
* PyTorch (1.1.0)
* torchvision (0.2.2)
* NumPy (1.14.3)
* SciPy (1.0.0)
* H5py (2.7.1)
* imageio (2.4.1)
* pandas (0.22.0)
* opencv-python (3.4.2)
* tqdm (4.19.8)
* matplotlib (3.0.2)

## License

This repository is licensed under the terms of the GNU AGPLv3 license.

## How to run the code

You need to first install conda in your environment. **Before running the code, please run `pip install -r requirements.txt` first.** Once you have installed all the dependencies, `run.sh` will automatically run the entire pipeline and save the prediction results in csv. Note that you need to first cd to the project directory and then execute `. ./run.sh`. When running the individual Python scripts, please include the path to this repository in your `PYTHONPATH`. 

We recommend running the code with a GPU. To run the code with CPU only, please change `DEVICE_TYPE` in run.sh to 'cpu'. 

## Data

## Reference