# Contextualizing hate speech models with explanations
Official code release for ACL 2020 paper [Contextualizing Hate Speech Classifiers with Post hoc Explanation](https://arxiv.org/abs/2005.02439)

## Requirements
```shell script
conda create -n expl-reg python==3.7.4
conda activate expl-reg
# modify CUDA version as yours
conda install pytorch=0.4.1 cuda90 -c pytorch
pip install nltk numpy scikit-learn scikit-image matplotlib torchtext
# requirements from pytorch-transformers
pip install tokenizers==0.0.11 boto3 filelock requests tqdm regex sentencepiece sacremoses
```

## Running experiments
See `scripts/` for shell scripts for running experiments. For example, to train a model on Gab dataset with SOC regularization, run
```shell scripts
chmod +x ./scripts/*
./scripts/gab_soc.sh
```

## Data
### Gab corpus
Gab corpus is available at https://osf.io/edua3/. Convert them into jsonl format and put train/dev/test.jsonl under `data/majority_gab_dataset_25k`, where each line is a json dict.
```
{"text_id":31287737,"Text":"How is that one post not illegal? He is calling for someone to commit a specific crime or he will do it himself. ","purity":0,"harm":0,"im":0,"cv":0,"ex":0,"degradation":0,"fairness":0,"hd":0,"mph":0,"loyalty":0,"care":0,"betrayal":0,"gen":0,"cheating":0,"subversion":0,"rel":0,"sxo":0,"rae":0,"nat":0,"pol":0,"authority":0,"vo":0,"idl":0}
```

### Stormfront corpus
The corpus is available at https://github.com/aitor-garcia-p/hate-speech-dataset. Convert them into tsv format and put train/dev/test.tsv under `data/white_supremacy`.
```
doc_id	text	is_hate
0	Somehow we 'll have our own Texas site , some day .	0
```

### NYT corpus
We construct an adversarial test
set of New York Times (NYT) articles that are
filtered to contain a balanced, random sample of the twenty-five group identifiers. However, we do not have rights to release the data. You may construct the dataset by yourself, and put test.tsv under `data/nyt_keyword_sample`.
```
,nyt_text,hate,keyword
9,"The object of much of the criticism by moderates and liberals is the austere, almost harsh brand of Islam dominant in Saudi Arabia, which, because of its oil wealth and custodianship of Islam's holiest places, Mecca and Medina, has enormous influence in the Muslim world. This branch of Islam is often called Wahhabism.",0,muslim
```