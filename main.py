import torch

from torch.utils.tensorboard import SummaryWriter

from config import Config
from utils import *
from trainer import Bert_Bilstm_Crf


def main():
    config = Config()
    set_logger(config)
    writer = SummaryWriter(log_dir=os.path.join(config.output_path, "visual"), comment="ner")

    if config.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(config.gradient_accumulation_steps))

    use_gpu = torch.cuda.is_available() and config.use_gpu
    device = torch.device('cuda' if use_gpu else 'cpu')
    config.device = device
    n_gpu = torch.cuda.device_count()
    logging.info(f"available device: {device}，count_gpu: {n_gpu}")

    config.label_list = get_labels(config)
    label2id = {label: i for i, label in enumerate(config.label_list)}
    id2label = {i: label for label, i in label2id.items()}
    logging.info("loading label2id and id2label dictionary successful!")

    # Bert_Bilstm_Crf模型的训练与测试
    trainer_bbc = Bert_Bilstm_Crf(config, device, use_gpu, n_gpu, writer, id2label)
    # trainer_bbc.train()
    trainer_bbc.test()

if __name__ == '__main__':
    main()
