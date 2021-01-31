---
layout: blog
title: CCF BDCI房地产问答匹配第二名解决方案
date: 2021-01-29 21:11:18
tags:
  - nlp
  - text classifier
  - qa match

categories:
  - competitions


---


## 摘要
房产行业聊天匹配问题本质上面是句子对的分类问题，即判断问题和回答是否匹配的文本分类问题。但是问答匹配不同的是除了依赖当前的回答，往往是在一个多伦对话场景下的产生背景，在本题目中通过对数据的分析，发现通过id问题下面的相邻id的问答具有上下文的关系，我们在设计模型的输入的时候引入了上下文的信息，使得模型能够学习到更多的因果关系。同时近年来基于transformer结构的预训练模型横扫各大nlp任务的SOTA，所以在模型结构方面我们也是采用了基于transformer的预训练语言模型结合任务做出了创新性的设计，取得了比较好的效果。我们的模型最终取得了线上第二的成绩。


## 关键词
文本分类、上下文信息、预训练模型、Transformer 

<!--more-->



## 1.任务简介
给定IM交流片段，片段包含一个客户问题以及随后的经纪人若干IM消息，从这些随后的经纪人消息中找出一个是对客户问题的回答。简单来说即是判断问题和回答这个句子对是不是匹配的。


## 2.预训练语言模型简介

### 2.1 BERT预训练模型
2018年google公司AI团队新发布的BERT模型[1]，在机器阅读理解顶级水平测试SQuAD1.1中表现出惊人的成绩：全部两个衡量指标上全面超越人类，并且还在11种不同NLP测试中创出最佳成绩，包括将GLUE基准推至80.4％（绝对改进7.6％），MultiNLI准确度达到86.7% （绝对改进率5.6％）等。BERT为NLP带来了里程碑式的改变，也是NLP领域近年来最重要的进展。    Bert base版本使用了12层的transformer [2] encoder层部分作为编码器来对文本输入提取语义特征，用空间向量来表示。Transformer通过self-attention机制使得相同的字在不同的语境下面有不同的空间向量表示。同时相比于传统的rnn,cnn特征提取层来说，突破了rnn不能并行计算的限制，相比cnn计算两个位置之间的关联所需的操作次数不随长度的增长。
      ![transformer](/image/transformer.png)
       图1：transformer网络结构Bert使用了两个预训练任务，来训练模型的权重。一个是mask language model， 即通过对输入的文本进行部分的随机替换成[MASK]字符，然后让模型来预测被[ MASK]的字符是什么，类似于完形填空任务。另外一个就是句子对分类任务，即判断相邻的句子是不是上下文关系。  

       ![bert_finetune](/image/bert_finetune.png)

       图2：bert预训练和finetune
       
       
       
### 2.2 BERT-WWM预训练模型
bert-wwm [3]是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。 简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个子词，在生成训练样本时，这些被分开的子词会随机被mask。 在全词Mask中，如果一个完整的词的部分WordPiece子词被mask，则同属该词的其他部分也会被mask，即全词Mask。



### 2.3 MACBERT预训练模型
macbert [4]作者针对Bert在做MLM预训练的时候使用的 [MASK]替换输入，但是在做别的下游任务finetune的时候是没有[MASK]输入的，这就导致了预训练任务输入和finetune输入的差异问题。不使用[MASK]token进行mask，因为在token微调阶段从未出现过[MASK]，而是通过使用同义词替换的方法进行替换。





## 3.解决方案

### 3.1 数据处理
通过数据分析，我们发现对同个相同的问题id下面，相邻的两个问题id部分存在上下文关系。 所以为了让模型能够学习到更多的上下文关系我们在模型的输入加入了相邻的文本的信息：
[CLS]问题[SEP]问答[START]相邻的上个回答[INSERT]相邻的下个回答[SEP]



### 3.2 模型结构

们设计了pytorch和tensorflow两个版本的模型。并且两个模型的单模型效果都在初赛上面进入了前5的成绩，由于两个模型不仅在框架上面，网络结构和输入上面也存在一定的差异，使得组队后，进行模型融合的时候，带来了比较大的提升。 两个模型融合直接进入了第二名的成绩和第一名差距也在一个千分点内。模型1:pytorch 版本，模型输入采用了下面的输入结构：[CLS]问题[SEP]问答[START]相邻的上个回答[INSERT]相邻的下个回答[SEP]。网络上面采用了bert等预训练模型作为特征提取层，由于bert不同层transformer提取出来的语义存在较大的差异，在不同语法上面侧重点不一样，我们设计了动态融合层，使用不同transformer层动态的加权方法来作为最后的表征。  
![pytorch model](/image/beike3.png)
图3：pytorch版本网络模型结构模型2: tensorflow版本。tf版模型将问题和回答按顺序拼接并使用“[SEP]”字符分割，并在问题前后插入“[unused1]”，回答前后按顺序插入“[unused2]”,“[unused3]”等字符后作为输入。模型可以大致分为编码层， 特征抽取层以及输出层。考虑到bert的不同层输出具有不同的语义信息，以及transformer对临近字符信息抽取能力较弱，编码层选择对bert各层输出加权融合，并输入bilstm强化对临近字符的抽取能力，最后选择bert各层的加权值以及bilstm的输出值的拼接向量作为编码层的输出。借鉴bert在信息抽取领域一些有效的结构，模型的特征抽取层被设计为：1.将[SEP],[unused1],[unused2]等特殊字符在编码层后的输出做为表征1；2.将回答，问题经过一层卷积和max-pooling的输出拼接值作为表征2；3. 将表征1，表征2拼接作为最终问题和回答的表征；4.最后将问题和回答的表征通过类似gate结构进行融合作为特征抽取层的输出。输出层简单的将上一层的输出输入dense加softmax得到是回答与不是回答的概率。
![tensorflow model](/image/beike4.png)
图4：tensorflow版本网络模型结构



### 3.3提分Trick

1.模型输入引入上下文信息提升巨大。  
2.使用对抗学习带来5个千分点的提升。  
3.在比赛数据语料上面继续做房产领域的的预训练带来5个千分点左右的提升。  
4.使用动态加权层来融合预训练模型不同层的输出带来3个左右的千分点的提升。



### 3.4 模型融合
我们pytorch和tensorflow两个版本的单模型都有着很好的效果在A榜能进入前5。后面由于竞争比较激烈，我们采用了多模型集成学习的方法，基于pytorch和tensorflow两个版本分别跑了bert-wwm和mac-bert两个预训练的交叉验证的结果作为特征，为了防止模型过拟合，采用了线形模型作为基模型。最终取得了A榜第二和B榜第二的成绩。和第一名仅仅差距在一个千分点内。



## 4.参考
[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In North American Association for Computational Linguistics (NAACL).

[2] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Pro- cessing Systems, pages 6000–6010. 
[3] Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, Ziqing Yang, Shijin Wang, and Guoping Hu. 2019a. Pre-training with whole word masking for chinese bert. arXiv preprint arXiv:1906.08101.
。
[4] Cui, Yiming  and Che, Wanxiang  and Liu, Ting  and Qin, Bing Wang, Shijin and Hu, Guoping . 2020. Revisiting Pre-Trained Models for chinese Natural Language Processing. EMNLP	


