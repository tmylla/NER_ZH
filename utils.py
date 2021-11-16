import os
import pickle
import logging

def set_logger(config):
    if not os.path.exists(config.log_path):
        os.mkdir(config.log_path)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=os.path.join(config.log_path, '{}.log'.format(config.model)),
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def load_pkl(fp):
    """加载pkl文件"""
    with open(fp, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pkl(data, fp):
    """保存pkl文件，数据序列化"""
    with open(fp, 'wb') as f:
        pickle.dump(data, f)


def load_file(fp: str, sep: str = None):
    """
    读取文件；
    若sep为None，按行读取，返回文件内容列表，格式为:[xxx,xxx,xxx,...]
    若不为None，按行读取分隔，返回文件内容列表，格式为: [[xxx,xxx],[xxx,xxx],...]
    """
    with open(fp, "r", encoding="utf-8") as f:
        lines = f.readlines()
        if sep:
            return [line.strip().split(sep) for line in lines]
        else:
            return lines

def get_labels(config):
    """读取训练数据获取标签"""
    label_pkl_path = os.path.join(config.base_path, "data/label_list.pkl")
    if os.path.exists(label_pkl_path):
        logging.info(f"loading labels info from {os.path.join(config.base_path, 'data')}")
        labels = load_pkl(label_pkl_path)
    else:
        logging.info(f"loading labels info from train file and dump in {os.path.join(config.base_path, 'data')}")
        tokens_list = load_file(config.train_file, sep=' ')
        labels = list(set([tokens[1] for tokens in tokens_list if len(tokens) == 2]))
        # 增加开始和结束的标志
        labels.extend(['<START>', '<END>'])
        save_pkl(labels, label_pkl_path)

    return labels