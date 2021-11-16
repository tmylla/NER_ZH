import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from torchcrf import CRF


class BERT_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config, need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size  # 768

        if need_birnn:
            self.need_birnn = need_birnn
            self.birnn = nn.LSTM(input_size=config.hidden_size, hidden_size=rnn_dim, num_layers=1, bidirectional=True,
                                 batch_first=True)
            out_dim = rnn_dim * 2

        self.hidden2tag = nn.Linear(in_features=out_dim, out_features=config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids, tags, token_type_ids=None, attention_mask=None):
        """
        :param input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
        :param token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
        :param attention_mask:     torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
        :param tags:
        :return:
        """
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # torch.Size([batch_size,seq_len,hidden_size])
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)  # (seq_length,batch_size,num_directions*hidden_size)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)  # [seq_length, batch_size, num_labels]
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())
        return loss

    def predict(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)
        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)
        return self.crf.decode(emissions, attention_mask.byte())
