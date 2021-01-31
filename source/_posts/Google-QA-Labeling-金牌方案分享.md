---
layout: blog
title: Google QA Labeling 金牌方案分享
date: 2020-03-12 20:34:37
tags:
  - nlp
  - kaggle
categories:
  - competitions
  
---


## 比赛背景与问题分析
[google qa labeling 比赛链接](https://www.kaggle.com/c/google-quest-challenge)
一个月前看到了kaggle上面google举办的一个nlp比赛，google quest Q&A labeling。就是根据问答对，预测30个不同对label值， 这些label值是人工给问答对打上对标签。标签值是0到1之间的数值评分。 这题既可以理解成回归的方法 也 可以理解成 分类的方法去做。  
## 方案总结
这次比赛中我们主要使用roberta large, roberta base, xlnet  也尝试过最新的google T5模型，发现效果不是很好。我们对文本清洗，模型结构对设计， loss修改，文本增强都进行了大量的参试。由于比赛public的测试集只有16%,所以我们并没有刻意的去拟合lb,也只是才用了较为简单的后处理方案。这使得我们在最终private开榜以后成功的升到了第5名，进入了奖金区。
![最终的排行榜](/image/googleqa.png)
<!--more-->

## 具体细节
由于个人比较懒这里就直接copy我在kaggle 论坛分享的这个比赛的我们队伍的解决方案。

First of all I want to thank my teammates here.I will briefly introduce our solution here.
models
1.Model structure。we design different models structure. We mainly refer to the solution of ccf internet sentiment analysis，concat different cls embedding . here it is the link BDCI2019-SENTIMENT-CLASSIFICATION
2.We found 30 labels through analysis, one is the question-related evaluation, and the other is the answer-related evaluation. In order to make the model learn better, we have designed the q model to remove the problem-related label and the a model to process the answer Related labels。 it is better than qa models.
3.different model test. roberta base >roberta large >xlnet base >bert base > t5 base.
Post-processing
Analysis and evaluation methods and competition data，we use 0,1 reseting way. it improve lb 0.05 or more.
Features
1.we want that our model learns features that are not only considered in text, so we add host and
category embeeding features annd other Statistical Features。 it improve both cv and lb about 0.005.
Text clean
1.We also did text cleaning to remove stop words and some symbols， it improve about 0.002
Stacking
1.Our Best Private model scored 0.42787 ,but we dont't select it. it is stacking by roberta large and roberta base and xlnet base.
blend.loc[:,targets] = roberta_large_oof_test.loc[:,targets].values*0.4+0.3*roberta_base_oof_test.loc[:,targets].values+\ xlnet_base_oof_test.loc[:,targets].values*0.3
stacking improve both cv and lb about 0.02 . it help much.
