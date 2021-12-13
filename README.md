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
### Gab Hate Corpus
The full Gab Hate Corpus (GHC) is available at https://osf.io/edua3/. Here, data files are prepared in train/dev/test.jsonl under `data/majority_gab_dataset_25k` in jsonl, where each line is a json dict.

```
{"text_id":31287737,"Text":"How is that one post not illegal? He is calling for someone to commit a specific crime or he will do it himself. ","im":0,"cv":0,"ex":0,"hd":0,"mph":0,"gen":0,"rel":0,"sxo":0,"rae":0,"nat":0,"pol":0,"vo":0,"idl":0}
```

The GHC can be cited using the following:

```
@misc{kennedy2020gab,
  title = {The {G}ab {H}ate {C}orpus: A {C}ollection of 27k {P}osts {A}nnotated for {H}ate {S}peech},
  url = {psyarxiv.com/hqjxn},
  doi = {10.31234/osf.io/hqjxn},
  publisher = {PsyArXiv},
  author = {Kennedy, Brendan and Atari, Mohammad and Mostafazadeh Davani, Aida and Yeh, Leigh and Omrani, Ali and Kim, Yehsong and Coombs Jr., Kris and Havaldar, Shreya and Portillo-Wightman, Gwenyth and Gonzalez, Elaine and Hoover, Joe and Azatian*, Aida and Cardenas*, Gabriel and Hussain*, Alyzeh and Lara*, Austin and Omary*, Adam and Park*, Christina and Wang*, Xin and Wijaya*, Clarisa and Zhang*, Yong and Meyerowitz, Beth and Dehghani, Morteza},
  year = {2020},
  month = feb
}
```

### Stormfront corpus
The corpus is available at https://github.com/aitor-garcia-p/hate-speech-dataset. Convert them into tsv format and put train/dev/test.tsv under `data/white_supremacy`.

```
doc_id	text	is_hate
0	Somehow we 'll have our own Texas site , some day .	0
```

The Stormfront dataset can be cited using the following:

```
@inproceedings{de2018hate,
   title={Hate {S}peech {D}ataset from a {W}hite {S}upremacy {F}orum},
   author={de Gibert, Ona and Perez, Naiara and Garc{\'\i}a-Pablos, Aitor and Cuadros, Montse},
   booktitle={Proceedings of the 2nd {W}orkshop on {A}busive {L}anguage {O}nline ({ALW2})},
   pages={11--20},
   year={2018}
 }
```

### NYT corpus
We construct an adversarial test set of New York Times (NYT) articles that are filtered to contain a balanced, random sample of the twenty-five group identifiers. Since we do not have rights to release the data, a similar test set can be constructed from a similar (e.g., news) domain, by using the filtering keywords in `data/identity.csv`. Place the resulting test.tsv file under `data/nyt_keyword_sample`.

```
,nyt_text,hate,keyword
9,"The object of much of the criticism by moderates and liberals is the austere, almost harsh brand of Islam dominant in Saudi Arabia, which, because of its oil wealth and custodianship of Islam's holiest places, Mecca and Medina, has enormous influence in the Muslim world. This branch of Islam is often called Wahhabism.",0,muslim
```


## Reference
If you find this code helpful, please use the following citation:

```
@inproceedings{kennedy2020contextualizing,
   author = {Kennedy*, Brendan and Jin*, Xisen and Mostafazadeh Davani, Aida and Dehghani, Morteza and Ren, Xiang},
   title = {Contextualizing {H}ate {S}peech {C}lassifiers with {P}ost-hoc {E}xplanation},
   year = {to appear},
   booktitle = {Proceedings of the 58th {A}nnual {M}eeting of the {A}ssociation for {C}omputational {L}inguistics}
} 
```
