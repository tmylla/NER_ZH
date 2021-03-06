

# 目录

[简介](#简介)

[安装](#安装)

[目录结构](#目录结构)

[模型介绍](#模型介绍)

[运行方式](#运行方式)

[参考](#参考)



# 简介

使用PyTorch实现中文NER模型，拟提供BERT_BiLSTM_CRF、BiLSTM_CRF、CRF和HMM四种模型，目前实现BERT_BiLSTM_CRF。



# 安装

```sh
tqdm==4.62.2
torch==1.8.2+cu102
transformers==4.11.3
torchcrf==1.1.0
```



# 目录结构

```python
NER_ZH
├── ckpts
│    ├── bert-base-chinese  # 预训练模型BERT
│    ├── bert_bilstm_crf    # 自己训练的bert_bilstm_crf模型
│    ├── ...
├── data  # 数据集
│    ├── test.txt
│    ├── train.txt
│    ├── ...
├── logs  # 训练日志
│    ├── bert_bilstm_crf.log  # bert_bilstm_crf训练测试日志
├── output  # 预测结果、可视乎结果等
├── config.py
├── dataloader.py
├── evaluator.py
├── main.py
├── models.py
├── trainer.py
├── utils.py
```



# 模型介绍

## BERT_BiLSTM_CRF

```python
"""
1. BERT:
	outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
    sequence_output = outputs[0]

    Inputs:
        input_ids:      torch.Size([batch_size,seq_len]), 代表输入实例的tensor张量
        token_type_ids: torch.Size([batch_size,seq_len]), 一个实例可以含有两个句子,相当于标记
        attention_mask: torch.Size([batch_size,seq_len]), 指定对哪些词进行self-Attention操作
    Out:
        sequence_output: torch.Size([batch_size,seq_len,hidden_size]), 输出序列
        pooled_output:   torch.Size([batch_size,hidden_size]), 对输出序列进行pool操作的结果
        (hidden_states): tuple, 13*torch.Size([batch_size,seq_len,hidden_size]), 隐藏层状态，取决于config的output_hidden_states
        (attentions):    tuple, 12*torch.Size([batch_size, 12, seq_len, seq_len]), 注意力层，取决于config中的output_attentions
        
       
2. BiLSTM:
	self.birnn = nn.LSTM(input_size=config.hidden_size, hidden_size=rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
	Args:
		input_size:    输入数据的特征维数
		hidden_size:   LSTM中隐层的维度
        num_layers:    循环神经网络的层数
        bias:          用不用偏置，default=True
        batch_first:   通常我们输入的数据shape=(batch_size,seq_length,input_size),而batch_first默认是False,需要将batch_size与seq_length调换
        dropout:       默认是0，代表不用dropout
        bidirectional: 默认是false，代表不用双向LSTM

	sequence_output, _ = self.birnn(sequence_output)
    Inputs:
		input:shape=(seq_length,batch_size,input_size)的张量
        (h_0,c_0): h_0.shape=(num_directions*num_layers, batch, hidden_size)，它包含了在当前这个batch_size中每个句子的初始隐藏状态；num_layers就是LSTM的层数，如果bidirectional=True,num_directions=2,否则就是１，表示只有一个方向，c_0和h_0的形状相同，它包含的是在当前这个batch_size中的每个句子的初始状态，h_0、c_0如果不提供，那么默认是０
		OutPuts:
        output:shape=(seq_length,batch_size,num_directions*hidden_size), 它包含LSTM的最后一层的输出特征(h_t)
        (h_n,c_n): h_n.shape=(num_directions*num_layers, batch, hidden_size), c_n与h_n形状相同, h_n包含的是句子的最后一个单词的隐藏状态；c_n包含的是句子的最后一个单词的细胞状态，所以它们都与句子的长度seq_length无关；output[-1]与h_n是相等的，因为output[-1]包含的正是batch_size个句子中每一个句子的最后一个单词的隐藏状态，注意LSTM中的隐藏状态其实就是输出，cell state细胞状态才是LSTM中一直隐藏的，记录着信息


3. 全连接层:
	self.hidden2tag = nn.Linear(in_features=out_dim, out_features=config.num_labels)
	Args:
		in_features:  输入的二维张量的大小，即输入的[batch_size, size]中的size
		out_features: 输出的二维张量的大小，即输出的二维张量的形状为[batch_size，output_size]，当然，它也代表了该全连接层的神经元个数

	释义:
		从输入输出的张量的shape角度来理解，相当于一个输入为[batch_size, in_features]的张量变换成了[batch_size, out_features]的输出张量


4. CRF:
    self.crf = CRF(num_tags=config.num_labels, batch_first=True)
    Args:
        num_tags:Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    loss = -1 * self.crf(emissions, tags, mask=input_mask.byte())
    Inputs:
        emissions (`~torch.Tensor`): Emission score tensor of size``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,``(batch_size, seq_length, num_tags)`` otherwise.
        tags (`~torch.LongTensor`): Sequence of tags tensor of size``(seq_length, batch_size)`` if ``batch_first`` is ``False``,``(batch_size, seq_length)`` otherwise.
        mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        reduction: Specifies  the reduction to apply to the output:``none|sum|mean|token_mean``. ``none``: no reduction will be applied; ``sum``: the output will be summed over batches; ``mean``: the output will be averaged over batches; ``token_mean``: the output will be averaged over tokens.

    Returns:
    `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if reduction is ``none``, ``()`` otherwise.
"""
```

## 模型评估

- **精确率P、召回率R以及F1值**

  $P=\frac{TP}{TP+FP}$

  $R=\frac{TP}{TP+FN}$

  $F1=\frac{2PR}{P+R}$

  - ```python
    '''
    通过'evaluator.py'计算每个标签的精确率、召回率和F1分数，输出如下格式：
               precision    recall  f1-score   support
            O     0.9999    0.9999    0.9999    150935
        I-ORG     0.9984    0.9991    0.9988      5640
        I-LOC     0.9968    0.9970    0.9969      4370
        B-ORG     0.9955    0.9985    0.9970      1327
        I-PER     1.0000    0.9995    0.9997      3845
        B-LOC     0.9990    0.9962    0.9976      2871
        B-PER     0.9995    1.0000    0.9997      1972
    avg/total     0.9997    0.9997    0.9997    170960
    '''
    ```
    
  - 由于标签中 “O”占非常大的比例，因此在计算指标时，采用两种方式：一是直接计算所有的指标，二是去掉“O”这个类别后再计算所有指标。修改`config.py`中的`self.remove_O = False/True`实现不同计算方式。
  
- **混淆矩阵**

  - ```python
    '''
    Confusion Matrix:
                O   I-ORG   I-LOC   B-ORG   I-PER   B-LOC   B-PER 
        O  150921       6       7       1       0       0       0 
    I-ORG       1    5635       0       4       0       0       0 
    I-LOC       8       2    4357       0       0       3       0 
    B-ORG       1       1       0    1325       0       0       0 
    I-PER       1       0       0       0    3843       0       1 
    B-LOC       3       0       7       1       0    2860       0 
    B-PER       0       0       0       0       0       0    1972 
    '''
    ```

  

# 运行方式

根据需要自行设置`config.py`文件参数，然后运行`main.py`文件即可。



# 参考

1. [bert_bilstm_crf_ner_pytorch](https://gitee.com/chenzhouwy/bert_bilstm_crf_ner_pytorch/tree/master)
2. [named_entity_recognition](https://github.com/luopeixiang/named_entity_recognition)

