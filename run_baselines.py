"""
Code for training TF-IDF linear classifiers and obtaining their linear coefficients
Note: CountVectorizer performs better than TF-IDF. Perhaps the dataset is biased in that
    long sequences are more likely to be offensive.
"""


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from loader import GabProcessor, WSProcessor, NytProcessor
from utils.config import configs
from bert.tokenization import BertTokenizer
import argparse
import numpy as np
import pickle, os
import random

def examples_to_bow(examples, tokenizer, max_seq_length):
    nw_file = args.nw_file
    nw_words= []
    if args.remove_nw:
        f = open(nw_file)
        nw_words = set([_.strip().split('\t')[0] for _ in f.readlines()])
        f.close()

    inputs, labels = [], []
    all_input_tokens = []
    vocab = tokenizer.vocab
    for example in examples:
        tokens = tokenizer.tokenize(example.text_a)

        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        tokens = ['[CLS]'] + tokens + ['[SEP]']

        if nw_words:
            tokens = list(filter(lambda x: x not in nw_words, tokens))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        all_input_tokens.append(tokens)
        bow = np.zeros(len(vocab))
        for tok in input_ids:
            bow[tok] = 1
        inputs.append(bow)
        labels.append(int(example.label))

    avg_length = sum([len(x) for x in all_input_tokens]) / len(all_input_tokens)
    #print(avg_length)
    #print(sum(labels) / len(labels))

    return inputs, labels, all_input_tokens

def fit_tfidf_model(dataset):
    if dataset == 'gab':
        data_processor = GabProcessor(configs)
    else: # dataset is 'ws'
        configs.data_dir = './data/white_supremacy/'
        data_processor = WSProcessor(configs)

    model = LogisticRegression()
    tokenizer = BertTokenizer.from_pretrained(configs.bert_model, do_lower_case=configs.do_lower_case)

    train_examples, val_examples = data_processor.get_train_examples(configs.data_dir), \
                                            data_processor.get_dev_examples(configs.data_dir)
    random.shuffle(train_examples)

    gab_processor = GabProcessor(configs)
    gab_test_examples = gab_processor.get_test_examples('./data/majority_gab_dataset_25k/')

    _, train_labels, train_tokens = examples_to_bow(train_examples, tokenizer, configs.max_seq_length)
    _, val_labels, val_tokens = examples_to_bow(val_examples, tokenizer, configs.max_seq_length)
    _, test_labels, test_tokens = examples_to_bow(gab_test_examples, tokenizer, configs.max_seq_length)

    train_docs, val_docs = [' '.join(x) for x in train_tokens], [' '.join(x) for x in val_tokens]

    # binary BOW vector performs better than tfidf
    #vectorizer = TfidfVectorizer(tokenizer=str.split)
    vectorizer = CountVectorizer(binary=True)

    X = vectorizer.fit_transform(train_docs)

    neg_weight = 0.125 if dataset == 'ws' else 0.1
    weights = [1 if x == 1 else neg_weight for x in train_labels]

    model.fit(X, train_labels, weights)

    X_val = vectorizer.transform(val_docs)

    pred_gab_val = model.predict(X_val)
    f1 = f1_score(val_labels, pred_gab_val)
    print('val f1: %f' % f1)

    test_docs = [' '.join(x) for x in test_tokens]
    X_test = vectorizer.transform(test_docs)
    pred_gab_test = model.predict(X_test)
    gab_f1 = f1_score(test_labels, pred_gab_test)
    gab_p, gab_r = precision_score(test_labels, pred_gab_test), recall_score(test_labels, pred_gab_test)

    print('Gab test f1: %f (%f, %f)' % (gab_f1, gab_p, gab_r))

    ws_processor, nyt_processor = WSProcessor(configs), NytProcessor(configs, subset=dataset == 'ws')
    ws_test_examples = ws_processor.get_test_examples('data/white_supremacy')
    _, test_labels, test_tokens = examples_to_bow(ws_test_examples, tokenizer, configs.max_seq_length)
    test_docs = [' '.join(x) for x in test_tokens]
    X_test = vectorizer.transform(test_docs)
    pred_ws_test = model.predict(X_test)
    ws_f1 = f1_score(test_labels, pred_ws_test)
    ws_p, ws_r = precision_score(test_labels, pred_ws_test), recall_score(test_labels, pred_ws_test)
    print('WS test f1: %f (%f, %f)' % (ws_f1, ws_p, ws_r))

    nyt_test_examples = nyt_processor.get_test_examples('data/nyt_keyword_sample')
    _, test_labels, test_tokens = examples_to_bow(nyt_test_examples, tokenizer, configs.max_seq_length)
    test_docs = [' '.join(x) for x in test_tokens]
    X_test = vectorizer.transform(test_docs)
    pred_nyt_test = model.predict(X_test)
    nyt_f1 = accuracy_score(test_labels, pred_nyt_test)
    print('Nyt test f1: %f' % nyt_f1)

    dump_coeff(model, vectorizer)
    return gab_f1, gab_p, gab_r, ws_f1, ws_p, ws_r, nyt_f1

def dump_coeff(model, vectorizer):
    """
    store the coefficients of each word to the file, each row notes for a word and its
    coefficient in the linear model.
    :param model:
    :param vectorizer:
    :return:
    """
    f = open(args.coeff_dump, 'w')
    coeff = model.coef_[0]
    id_to_token = {}
    for k, v in vectorizer.vocabulary_.items():
        id_to_token[v] = k

    tuples = []
    for i in range(len(coeff)):
        tuples.append((id_to_token[i], coeff[i]))
        #f.write('%s\t%.6f\n' % (id_to_token[i], coeff[i]))

    tuples.sort(key=lambda x: -x[1])
    for tup in tuples:
        f.write('%s\t%.6f\n' % tup)

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['ws','gab'])
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument('--model_name')
    parser.add_argument('--max_seq_length', default=128)
    parser.add_argument('--remove_nw', action='store_true')

    # where to dump coeffs
    parser.add_argument('--coeff_dump', default='evals/coeff.csv')

    # where to load neutral words
    parser.add_argument('--nw_file', default='data/identity.csv')

    args = parser.parse_args()
    configs.update(args)

    results = []
    for seed in range(1):
        random.seed(seed)
        np.random.seed(seed)
        print('seed: %d' % seed)
        result = fit_tfidf_model(dataset=args.dataset)
        results.append(result)