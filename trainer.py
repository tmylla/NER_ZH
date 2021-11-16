import os
import torch
import logging

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer, BertConfig

from utils import *
from dataloader import NERDataset
from models import BERT_BiLSTM_CRF
from evaluator import Metrics


class Bert_Bilstm_Crf():
    def __init__(self, config, device, use_gpu, n_gpu, writer, id2label):
        self.config = config
        self.device = device
        self.use_gpu = use_gpu
        self.n_gpu = n_gpu
        self.writer = writer
        self.id2label = id2label
        self.tokenizer = BertTokenizer.from_pretrained(config.model_name_or_path,
                                                  do_lower_case=config.do_lower_case)
        bert_config = BertConfig.from_pretrained(config.model_name_or_path, num_labels=len(config.label_list))
        self.model = BERT_BiLSTM_CRF.from_pretrained(config.model_name_or_path, config=bert_config,
                                                need_birnn=config.need_birnn, rnn_dim=config.rnn_dim)
        self.model.to(device)
        logging.info("loading tokenizer、bert_config and bert_bilstm_crf model successful!")

    def train(self):
        if self.use_gpu and self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)

        logging.info("starting load train data and data_loader...")
        dataset = NERDataset(self.config, self.tokenizer, mode='train')
        dataloader = DataLoader(dataset, self.config.batch_size, shuffle=True)
        logging.info("loading train data_set and data_loader successful!")

        # 初始化模型参数优化器
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)

        # 初始化学习率优化器
        t_total = len(dataloader) // self.config.gradient_accumulation_steps * self.config.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.config.warmup_steps,
                                                    num_training_steps=t_total)
        logging.info("loading AdamW optimizer、Warmup LinearSchedule and calculate optimizer parameter successful!")

        logging.info("====================== Running training ======================")
        logging.info(
            f"Num Examples:  {len(dataset)}, Num Batch Step: {len(dataloader)}, "
            f"Num Epochs: {self.config.num_train_epochs}, Num scheduler steps：{t_total}")

        # 启用 BatchNormalization 和 Dropout
        self.model.train()
        global_step, tr_loss, logging_loss, best_f1 = 0, 0.0, 0.0, 0.0
        for epoch in range(int(self.config.num_train_epochs)):
            # model.train()
            for batch, batch_data in enumerate(tqdm(dataloader, desc="Train_DataLoader")):
                # input_ids = torch.tensor(batch_data['input_ids'], dtype=torch.long)
                # token_type_ids = torch.tensor(batch_data['token_type_ids'], dtype=torch.long)
                # attention_mask = torch.tensor(batch_data['attention_mask'], dtype=torch.long)
                # label_ids = torch.tensor(batch_data['label_ids'], dtype=torch.long)

                batch_data = tuple(torch.stack(batch_data[k]).T.to(self.device) for k in batch_data.keys())
                input_ids, token_type_ids, attention_mask, label_ids = batch_data
                outputs = self.model(input_ids, label_ids, token_type_ids, attention_mask)
                loss = outputs

                if self.use_gpu and self.n_gpu > 1:
                    loss = loss.mean()

                if self.config.gradient_accumulation_steps > 1:
                    loss = loss / self.config.gradient_accumulation_steps

                logging.info(f"Epoch: {epoch}/{int(self.config.num_train_epochs)}\tBatch: {batch}/{len(dataloader)}\tLoss:{loss}")
                # 反向传播
                loss.backward()
                tr_loss += loss.item()

                # 优化器_模型参数的总更新次数，和上面的t_total对应
                if (batch + 1) % self.config.gradient_accumulation_steps == 0:
                    # 更新参数
                    optimizer.step()
                    scheduler.step()
                    # 梯度清零
                    self.model.zero_grad()
                    global_step += 1

                    if self.config.logging_steps > 0 and global_step % self.config.logging_steps == 0:
                        tr_loss_avg = (tr_loss - logging_loss) / self.config.logging_steps
                        self.writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                        logging_loss = tr_loss

            if self.config.do_eval:
                logging.info("====================== Running Eval ======================")
                eval_data = NERDataset(self.config, self.tokenizer, mode="eval")

                avg_metrics, cal_indicators, eval_sens = self.evaluate(
                    self.config, self.tokenizer, eval_data, self.model, self.id2label, self.device, tqdm_desc="Eval_DataLoader")
                f1_score = avg_metrics['f1_score']
                self.writer.add_scalar("Eval/precision", avg_metrics['precision'], epoch)
                self.writer.add_scalar("Eval/recall", avg_metrics['recall'], epoch)
                self.writer.add_scalar("Eval/f1_score", avg_metrics['f1_score'], epoch)

                # save the best performs model
                if f1_score > best_f1:
                    logging.info(f"******** the best f1 is {f1_score}, save model !!! ********")
                    best_f1 = f1_score
                    # Take care of distributed/parallel training
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    model_to_save.save_pretrained(self.config.trained_model_path)
                    self.tokenizer.save_pretrained(self.config.trained_model_path)
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    model_to_save.save_pretrained(os.path.join(self.config.trained_model_path, 'checkpoints'))
                    self.tokenizer.save_pretrained(os.path.join(self.config.trained_model_path, 'checkpoints'))

            # # （如果config.do_eval=False，注释以下模型断点保存步骤）
            # # 数据集过大，需要分阶段、分时训练时每隔一段时间保存checkpoints
            # if (epoch + 1) % self.config.ckpts_epoch == 0:
            #     model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
            #     model_to_save.save_pretrained(os.path.join(self.config.trained_model_path, 'checkpoints'))
            #     self.tokenizer.save_pretrained(os.path.join(self.config.trained_model_path, 'checkpoints'))

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(os.path.join(self.config.trained_model_path, 'checkpoints'))
        self.tokenizer.save_pretrained(os.path.join(self.config.trained_model_path, 'checkpoints'))

        # torch.save(self.config, os.path.join(self.config.trained_model_path, 'training_config.bin'))
        # torch.save(self.model, os.path.join(self.config.trained_model_path, 'ner_model.ckpt'))
        # logging.info("training_args.bin and ner_model.ckpt save successful!")

        self.writer.close()
        logging.info("NER model training successful!!!")

    @staticmethod
    def evaluate(config, tokenizer, dataset, model, id2label, device, tqdm_desc):
        sampler = SequentialSampler(dataset)
        data_loader = DataLoader(dataset, sampler=sampler, batch_size=config.batch_size)
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        model.eval()

        id2label[-1] = 'NULL'  # 解码临时添加
        ori_tokens = [tokenizer.decode(tdt['input_ids']).split(" ") for tdt in dataset]
        ori_labels = [[id2label[idx] for idx in tdt['label_ids']] for tdt in dataset]
        pred_labels = []

        for b_i, batch_data in enumerate(tqdm(data_loader, desc=tqdm_desc)):
            batch_data = tuple(torch.stack(batch_data[k]).T.to(device) for k in batch_data.keys())
            input_ids, token_type_ids, attention_mask, label_ids = batch_data

            with torch.no_grad():
                logits = model.predict(input_ids, token_type_ids, attention_mask)

            for logit in logits:
                pred_labels.append([id2label[idx] for idx in logit])

        assert len(pred_labels) == len(ori_tokens) == len(ori_labels)
        eval_sens = []
        for ori_token, ori_label, pred_label in zip(ori_tokens, ori_labels, pred_labels):
            sen_tll = []
            for ot, ol, pl in zip(ori_token, ori_label, pred_label):
                if ot in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                sen_tll.append((ot, ol, pl))
            eval_sens.append(sen_tll)

        golden_tags = [[ttl[1] for ttl in sen] for sen in eval_sens]
        predict_tags = [[ttl[2] for ttl in sen] for sen in eval_sens]
        cal_indicators = Metrics(golden_tags, predict_tags, remove_O=config.remove_O)
        avg_metrics = cal_indicators.cal_avg_metrics()  # avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1_score']

        return avg_metrics, cal_indicators, eval_sens


    def test(self):
        logging.info("====================== Running test ======================")
        dataset = NERDataset(self.config, self.tokenizer, mode='test')
        avg_metrics, cal_indicators, eval_sens = self.evaluate(
            self.config, self.tokenizer, dataset, self.model, self.id2label, self.device, tqdm_desc="Test_DataLoader")

        cal_indicators.report_scores()  # avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1_score']
        cal_indicators.report_confusion_matrix()
        # 将测试结果写入本地
        with open(os.path.join(self.config.output_path, "token_labels_test.txt"), "w", encoding="utf-8") as f:
            for sen in eval_sens:
                for ttl in sen:
                    f.write(f"{ttl[0]}\t{ttl[1]}\t{ttl[2]}\n")
                f.write("\n")

        # sampler = SequentialSampler(dataset)
        # data_loader = DataLoader(dataset, sampler=sampler, batch_size=self.config.batch_size)
        # self.model.eval()
        #
        # id2label = self.id2label
        # id2label[-1] = 'NULL'  # 解码临时添加
        # ori_tokens = [self.tokenizer.decode(tdt['input_ids']).split(" ") for tdt in dataset]
        # ori_labels = [[id2label[idx] for idx in tdt['label_ids']] for tdt in dataset]
        # pred_labels = []
        #
        # for b_i, batch_data in enumerate(tqdm(data_loader, desc="Test_DataLoader")):
        #     batch_data = tuple(torch.stack(batch_data[k]).T.to(self.device) for k in batch_data.keys())
        #     input_ids, token_type_ids, attention_mask, label_ids = batch_data
        #
        #     with torch.no_grad():
        #         logits = self.model.predict(input_ids, token_type_ids, attention_mask)
        #
        #     for logit in logits:
        #         pred_label = []
        #         for idx in logit:
        #             pred_label.append(id2label[idx])
        #         pred_labels.append(pred_label)
        #
        # assert len(pred_labels) == len(ori_tokens) == len(ori_labels)
        # eval_sens = []
        # for ori_token, ori_label, pred_label in zip(ori_tokens, ori_labels, pred_labels):
        #     sen_tll = []
        #     for ot, ol, pl in zip(ori_token, ori_label, pred_label):
        #         if ot in ["[CLS]", "[SEP]", "[PAD]"]:
        #             continue
        #         sen_tll.append((ot, ol, pl))
        #     eval_sens.append(sen_tll)
        #
        # golden_tags = [[ttl[1] for ttl in sen] for sen in eval_sens]
        # predict_tags = [[ttl[2] for ttl in sen] for sen in eval_sens]
        # cal_indicators = Metrics(golden_tags, predict_tags, remove_O=self.config.remove_O)
        # avg_metrics = cal_indicators.cal_avg_metrics()






