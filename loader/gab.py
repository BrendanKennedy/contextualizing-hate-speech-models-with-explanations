from .common import *
from torch.utils.data import DataLoader, Dataset
import torch
import random

class GabProcessor(DataProcessor):
    """
    Data processor using DataProcessor class provided by BERT
    """
    def __init__(self, configs, tokenizer=None):
        super().__init__()
        self.data_dir = configs.data_dir
        self.label_groups = configs.label_groups
        self.tokenizer = tokenizer
        self.max_seq_length = configs.max_seq_length
        self.configs = configs
        self.remove_nw = configs.remove_nw

    def _get_neutral_word_ids(self):
        f = open(self.configs.neutral_words_file)
        neutral_words = []
        neutral_words_ids = set()
        for line in f.readlines():
            word = line.strip().split('\t')[0]
            canonical = self.tokenizer.tokenize(word)
            if len(canonical) > 1:
                canonical.sort(key=lambda x: -len(x))
                print(canonical)
            word = canonical[0]
            neutral_words.append(word)
            neutral_words_ids.add(self.tokenizer.vocab[word])
        self.neutral_words = neutral_words
        self.neutral_words_ids = neutral_words_ids
        assert neutral_words


    def _create_examples(self, data_dir, split, label=None):
        """
        Create a list of InputExample, where .text_a is raw text and .label is specified
        as configs.label_groups
        :param data_dir:
        :param split:
        :param label:
        :return:
        """
        f = open(os.path.join(data_dir, '%s.jsonl' % split))
        examples = []
        for i, line in enumerate(f.readlines()):
            data = json.loads(line)
            example = InputExample(text_a=data['Text'], guid='%s-%s' % (split, i))

            for j, label_group in enumerate(self.label_groups):
                tn = 0
                for key in label_group:
                    tn += int(data[key])
                setattr(example, 'label' if j == 0 else 'label_%d' % j, 1 if tn else 0)
            if label is None or example.label == label:
                examples.append(example)

        return examples

    def get_train_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'train', label)

    def get_dev_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'dev', label)

    def get_test_examples(self, data_dir, label=None):
        return self._create_examples(data_dir, 'test', label)

    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError

    def get_labels(self):
        return [0,1]

    def get_features(self, split):
        """
        Return a list of dict, where each dict contains features to be fed into the BERT model
        for each instance. ['text'] is a LongTensor of length configs.max_seq_length, either truncated
        or padded with 0 to match this length.
        :param split: 'train' or 'dev'
        :return:
        """

        neutral_word_ids = []
        if self.configs.remove_nw:
            neutral_word_ids = self._get_neutral_word_ids()

        examples = self._create_examples(self.data_dir, split)
        features = []
        for example in examples:
            tokens = self.tokenizer.tokenize(example.text_a)
            if len(tokens) > self.max_seq_length - 2:
                tokens = tokens[:(self.max_seq_length - 2)]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            if self.configs.remove_nw:
                input_ids = list(filter(lambda x: x not in neutral_word_ids, input_ids))
            length = len(input_ids)
            padding = [0] * (self.max_seq_length - length)
            input_ids += padding
            input_ids = torch.LongTensor(input_ids)
            features.append({'text': input_ids, 'length': length})
        return features

    def get_dataloader(self, split, batch_size=1):
        """
        return a torch.utils.DataLoader instance, mainly used for training the language model.
        :param split:
        :param batch_size:
        :return:
        """
        features = self.get_features(split)
        dataset = GabDataset(features)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=dotdict_collate)
        return dataloader

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer


class GabDataset(Dataset):
    """
    torch.utils.Dataset instance for building torch.utils.DataLoader, for training the language model.
    """
    def __init__(self, features):
        super().__init__()
        self.features = features

    def __getitem__(self, item):
        return self.features[item]

    def __len__(self):
        return len(self.features)
