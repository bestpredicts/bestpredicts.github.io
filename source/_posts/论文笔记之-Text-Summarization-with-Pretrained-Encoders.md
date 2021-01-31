---
layout: blog
title: 论文笔记之 Text Summarization with Pretrained Encoders
date: 2020-10-12 01:23:12
tags:
  - nlp
  - text summarization
categories:
  - text summarization

---

## Abstract：
像bert这种基于双向transformer作为encoder的预训练模型最近几年已经广泛用来各种的自然语言的下游任务。这篇论文主要提出一种基于bert的框架来做抽取和生成两种文本摘要方法。  
提出了一种基于文档级别的句子编码方案来获得每个句子的表征。  
通过把句子拼接到一起的方法。  
针对生成式摘要主要是通过表格学习率来分别针对encoder和decoder来进行finetune.实验表明这种分两步进行finetune的方法提高了生成摘要模型的效果。  


## 1. Introduction  
预训练语言模型已经在很大nlp任务上面取得了SOTA的效果。大多数情况下，预训练模型主要被用来作为各种nlp任务的句子或者段落的编码器，包括各种分类任务。在本文中，研究了预训练语言模型在文本摘要中的应用,不同于别的任务文本摘要需要超越个别词和句子更广泛的自然语言理解能力。目标是将文档进行压缩成更短的文本且保留文章的主要语义。 
对于生成式摘要来说需要语言生成模型包含新颖的单词和短语摘要。对于抽取式任务来说通常当作对句子进行二分类任务。  
提出了一种新颖的基于Bert的文档编码器，使其能够获得文档和句子的表征，具体做法是通过拼接句子间的transformer层用来作为句子的表征用来作为抽取任务。对于生成模型来说，采取的是encoder-decoder 的结构。使用预训练模型bert作为encoder,使用随机初始化的decoer.设计了新的训练方法，针对encoder和decoer使用不同的optimizer.受之前工作的启发，通过结合抽取任务结果可以用来提升生成模型效果，提出了两步优化的方法，针对encoder，做了两次finetune，第一次使用抽取任务来做finetune,然后再用到生成任务上面来。  

<!--more-->


## 2. Background  

### 2.1 Pretrained Language Models  
bert可以通过向句子开头插入[CLS] token,cls向量通常用来聚合表示整个句子的信息。[SEP] token通常用来插入到两个句子之间，用来表示一个句子的结尾。一篇可以用[w1,...,wn]个token来表示，每个token由 token embedding 表示每个token的空间向量，segmentation embedding用来切分表示两个句子,position embedding 用来表示每个token在句子中的位置。每个token的最终向量由3个向量相加得到。   
![bertsum](/image/bertsum.png)



### 2.2 Extractive Summarization  

抽取式摘要通常被当作句子分类任务来解决，从文本中抽取出重要的句子。  SUMMARUNNER (Nal- lapati et al., 2017) 是最早的通过rnn对句子进行编码，REFRESH (Narayan et al., 2018b)是通过强化学习的方法 ..... 

### 2.3 Abstractive Summarization  
抽取式摘要通常被当作seq2seq的问题，输入为 x = [x1, ..., xn]，输出为 y = [y1, ..., ym] ，以自回归的方式，来求解条件概率p(y1, ..., ym|x1, ..., xn)。Rush et al. (2015) and Nallapati et al. (2016) 是第一个将encoder-decoder的方法运用到文本摘要问题上的，See et al. (2017)通过pointer- generator network网络提高了模型效果，



## 3 Fine-tuning BERT for Summarization

### 3.1 Summarization Encoder  
尽管bert已经应用到很多nlp任务上面了，但是bert没法直接运用到文本摘要任务上。在bert中尽管segmentation embedding能够表示不同的句子，但是只是用在句子对输入的时候。针对多句子输入，获得不同句子的向量表示，设计了下面的框架
![bertsum](/image/bertsum.png)
 为了能获得每个句子的独立表示向量，在每个句子的开头插入[cls]令牌，针对segment embedding对相邻的句子使用不同的向量，文章中表述的是Senti. 当i是奇数或者偶数时候使用EA或者EB来表示。针对bert输入长度不能大于512的限制，针对positation embedding 大于512的部分进行随机随机初始化。  
 
 
 ### 3.2 Extractive Summarization
通过上面的插入[cls]的方法，可以将[CLS]的token向量来表示第i个句子，然后实验了bert的第1，2，3层transformers输出向量，实验是当取第二层transformers向量输出，后面接上sigmoid 二分类交叉熵，模型取得最好的效果。

### 3.3 Abstractive Summarization


## 4. Experimental Setup
## 5. Results


## 优点与缺点

### 优点
针对在输入多个句子的时候，设计在每个句子前面再插入[cls]的方法来表示每个句子向量，设计的很巧妙，且简单实用。针对bert输入长度的限制，提出了随机初始化大于512的位置向量的方法。

### 缺点





