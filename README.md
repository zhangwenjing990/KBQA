# NLPCC2016KBQA
##  项目背景

​	本项目数据来源于NLPCC会议的Open Domain Question Answering比赛，比赛描述连接：<http://tcci.ccf.org.cn/conference/2016/dldoc/evagline2.pdf>。

​	在项目开始之前，仔细研读了该比赛中冠军队伍的模型，并以该架构为baseline，尝试使用不同的模型进行优化。

###  任务描述

​	该任务的目标是基于给定KB(knowledge base)，构建自然语言问答系统(KBQA)。对于每个问题，从KBQA中选择相应实体，作为答案。

​	给定KB为三元组形式，即：subkect-predicate-object

###  数据描述

​	数据集包含一个知识图谱，一个训练集和一个测试集。知识图谱由三元组构成，形式如下：
​        ![img](https://uploader.shimo.im/f/EiAxpYKcnI4xIwyh.png!thumbnail)
​       

​	在训练集中，每个问题包含一个最佳答案，如果答案有多个组成元素，用'\t'隔开，形式如下：
​        ![img](https://uploader.shimo.im/f/Sepk2eLTmMMOoSaS.png!thumbnail)
​       测试集只有问题，需要我们预测答案。

###  评价指标

​	KBQA系统的质量由MRR, Accuracy@N, 和 Averaged F1进行评估：1，Mean Reciprocal Rank (MRR)
​        ![img](https://uploader.shimo.im/f/Os8RPQZA4aB7s8kJ.png!thumbnail)
​      |Q|表示评测集中的问题总数，rank_i表示在问题Q_i的答案集C_i中，最佳答案出现的位置。如果C_i不包含最佳答案，1/rank_i为0 2，Accuracy@N
​        ![img](https://uploader.shimo.im/f/5wR2AwwkpkWQhPFw.png!thumbnail)
​      当C_i至少包含A_i中的一个答案时，delta_i为1，否则为03，Averaged F1
​        ![img](https://uploader.shimo.im/f/TbVMA97NslIRnUwT.png!thumbnail)
​      如果C_i为空，或者C_i与A_i（最佳答案）不相交，F_i置为0，否则用下式计算：
​        ![img](https://uploader.shimo.im/f/dhnm4BSd8TJKX2WX.png!thumbnail)

其中，#(C_i, A_i)为C_i与A_i中共同出现的答案的个数。|C_i|与|A_i|表示C_i与A_i各自的答案数目。

## baseline模型解读

​	论文地址：<https://link.springer.com/chapter/10.1007%2F978-3-319-50496-4_65>论文是2016年比赛冠军文章，下面我们结合具体代码实现，解读该模型模型架构：
​        ![img](https://uploader.shimo.im/f/KbXtnbKd6yFAg5Gx.png!thumbnail)

     ###  Data Cleaning

​	清洗原始KB数据，并生成字典类型的KB实体。

​	在原始KB中，有些predicate存在特殊符号（’‘-’，‘•’）和html标签等，作者使用正则表达式去除了这些特殊符号；此外，在subject中，也存在一些特殊字符，如（’‘-’，‘•’，[ ]，《》，（）以及字母大写），与predicate不同的是，这些特殊符号可能会对object起到决定性作用

​	由于KB数据量非常大，为了节省查找时间，作者将KB转换为字典类型。形式如下：

`{'加佐拉': [{'Gazzola': '意大利', '地区': '艾米利亚-罗马涅大区', '省份': '皮亚琴察省', '人口（2009）': '2,018', '密度': '46/平方公里（120/平方英里）', '时区': 'CET（UTC+1）', '夏令时': 'CEST（UTC+2）', '邮编': '29010'}], '诺水河镇': [{'所属国': '中国', '所属市': '巴山市', '诺水河镇': '中华人民共和国', '上级行政区': '通江县', '行政区类型': '镇', '行政区划代码': '511921113', '村级区划单位数': '25', '社区数': '1', '行政村数': '24'}]}`

###  Core Question Extraction

​	原始训练集和测试集没有给出问题所在的三元组，为了寻找问题与KB之间的联系，作者首先找到每个问题下，相关联的若干候选三元组，然后根据规则将置信度最高的三元组作为该问题-答案对应的三元组，处理后的训练和测试数据形式如下：
​        ![img](https://uploader.shimo.im/f/zOTrXTm2RAmUb0Ar.png!thumbnail)
​      可以看到，每个问题和答案之间多了一行对应的三元组

###  Pattern

​	有了三元组之后，作者进一步对训练数据进行处理，将每个问题中的subject替换为字符 '(SUB)'，然后关联与主题对应的predicate，生成问答模式，并记录训练集中相同问答模式出现的次数。

​	问答模式出现的次数越多，表明它们关联性越强，predicate作为该问题谓词的概率越高，因此作者将问答模式出现的次数作为谓词评分系统的一个特征。

​	抽取出的问答模式结构如下：
​        ![img](https://uploader.shimo.im/f/MeybxwAlNNPR7ZOT.png!thumbnail)
​      

###  Entity Linking

​	作者将KB中的每一个subject与问题相匹配，如果问题中存在subject字符串，则将该subject和subject对应的若干predicate:object作为问题的潜在实体，生成Topic Entity Linking

###  Predicate Scoring

​	对问题进行Entity Linking之后，会生成若干subject和predicate，为了确定最佳predicate，使用如下公式：
​        ![img](https://uploader.shimo.im/f/5HlW1uhcIKsGSlfH.png!thumbnail)
​      其中，wp_i表示predicate中的第i个单词,wq_j表示问题中的第j个单词，lp_i是wp_i的长度。Sim是两个单词的语义相似度，作者采用的是词向量的余弦相似度。

​	该模块得到predicate_score，用于ranking。

###  Ranking

​	排序采用线性加权平均的方式，特征包括subject_score，predicate_score，answer_pattern_score，详见代码



## 模型构建与优化



## 思考总结

