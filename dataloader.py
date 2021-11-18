import os
import logging
import torch
from torch.utils.data import Dataset, TensorDataset

# from config import Config
from utils import load_pkl, save_pkl, load_file

class InputData(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, token_type_ids, attention_mask, label_id):
        """
        :param input_ids:       单词在词典中的编码
        :param attention_mask:  指定 对哪些词 进行self-Attention操作
        :param token_type_ids:  区分两个句子的编码（上句全为0，下句全为1）
        :param label_id:        标签的id
        """
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_id = label_id


class NERDataset(Dataset):
    def __init__(self, config, tokenizer, mode="train"):
        # text: a list of words, all text from the training dataset
        super(NERDataset, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        if mode == "train":
            self.file_path = config.train_file
        elif mode == "test":
            self.file_path = config.test_file
        elif mode == "eval":
            self.file_path = config.dev_file
        else:
            raise ValueError("mode must be one of train, or test")

        self.tdt_data = self.get_data()
        self.len = len(self.tdt_data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        对指定数据集进行预处理，进一步封装数据，包括:
        tdt_data：[InputData(guid=index, text=text, label=label)]
        feature：BatchEncoding( input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask,
                                label_id=label_ids)
        data_f： 处理完成的数据集, TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)
        """
        label_map = {label: i for i, label in enumerate(self.config.label_list)}
        max_seq_length = self.config.max_seq_length

        data = self.tdt_data[idx]
        data_text_list = data.text.split(" ")
        data_label_list = data.label.split(" ")
        assert len(data_text_list) == len(data_label_list)

        features = self.tokenizer(''.join(data_text_list), padding='max_length', max_length=max_seq_length, truncation=True)
        label_ids = [label_map[label] for label in data_label_list]
        label_ids = [label_map["<START>"]] + label_ids + [label_map["<END>"]]
        while len(label_ids) < max_seq_length:
            label_ids.append(-1)
        features.data['label_ids'] = label_ids

        return features


    def read_file(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            lines, words, labels = [], [], []
            for line in f.readlines():
                contends = line.strip()
                tokens = line.strip().split()
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label, word = [], []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words, labels = [], []
        return lines


    def get_data(self):
        '''数据预处理并返回相关数据'''
        lines = self.read_file()
        tdt_data = []
        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            word_piece = self.word_piece_bool(text)
            if word_piece:
                continue
            label = line[0]
            tdt_data.append(InputData(guid=guid, text=text, label=label))

        return tdt_data


    def word_piece_bool(self, text):
        word_piece = False
        data_text_list = text.split(' ')
        for i, word in enumerate(data_text_list):
            # 防止wordPiece情况出现，不过貌似不会
            token = self.tokenizer.tokenize(word)
            # 单个字符表示不会出现wordPiece
            if len(token) != 1:
                word_piece = True

        return word_piece


    @staticmethod
    def convert_data_to_features(self, tdt_data):
        """
        对输入数据进行特征转换
        例如:
            guid: 0
            tokens: [CLS] 王 辉 生 前 驾 驶 机 械 洒 药 消 毒 9 0 后 王 辉 ， 2 0 1 0 年 1 2 月 参 军 ， 2 0 1 5 年 1 2 月 退 伍 后 ， 先 是 应 聘 当 辅 警 ， 后 来 在 父 亲 成 立 的 扶 风 恒 盛 科 [SEP]
            input_ids: 101 4374 6778 4495 1184 7730 7724 3322 3462 3818 5790 3867 3681 130 121 1400 4374 6778 8024 123 121 122 121 2399 122 123 3299 1346 1092 8024 123 121 122 126 2399 122 123 3299 6842 824 1400 8024 1044 3221 2418 5470 2496 6774 6356 8024 1400 3341 1762 4266 779 2768 4989 4638 2820 7599 2608 4670 4906 102
            token_type_ids: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
            attention_mask: 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
            label_ids: 2 5 3 2 2 2 2 2 2 2 2 2 2 4 11 11 5 3 2 4 11 11 11 11 11 11 11 2 2 2 4 11 11 11 11 11 11 11 2 2 2 2 2 2 2 2 2 0 14 2 2 2 2 2 2 2 2 2 12 7 7 7 7 2
        """
        label_map = {label: i for i, label in enumerate(self.config.label_list)}
        max_seq_length = self.config.max_seq_length

        features = []
        for data in tdt_data:
            data_text_list = data.text.split(" ")
            data_label_list = data.label.split(" ")
            assert len(data_text_list) == len(data_label_list)

            tokens, labels, ori_tokens = [], [], []
            word_piece = False
            for i, word in enumerate(data_text_list):
                # 防止wordPiece情况出现，不过貌似不会
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                label = data_label_list[i]
                ori_tokens.append(word)
                # 单个字符不会出现wordPiece
                if len(token) == 1:
                    labels.append(label)
                else:
                    word_piece = True

            if word_piece:
                logging.info("Error tokens!!! skip this lines, the content is: %s" % " ".join(data_text_list))
                continue

            assert len(tokens) == len(ori_tokens)

            # feature = self.tokenizer(''.join(tokens), padding='max_length', max_length=max_seq_length, truncation=True)
            # label_ids = [label_map[label] for label in labels]
            # label_ids = [label_map["<START>"]] + label_ids + [label_map["<END>"]]
            # while len(label_ids) < max_seq_length:
            #     label_ids.append(-1)
            # feature.data['label_ids'] = label_ids
            # features.append(feature)

            if len(tokens) >= max_seq_length - 1:
                # -2的原因是因为序列需要加一个句首和句尾标志
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]

            label_ids = [label_map[label] for label in labels]
            new_tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(new_tokens)
            token_type_ids = [0] * len(input_ids)
            attention_mask = [1] * len(input_ids)
            label_ids = [label_map["<START>"]] + label_ids + [label_map["<END>"]]

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                attention_mask.append(0)
                token_type_ids.append(0)
                label_ids.append(0)

            features.append(InputFeatures(input_ids=input_ids,
                                          token_type_ids=token_type_ids,
                                          attention_mask=attention_mask,
                                          label_id=label_ids))
        return features
