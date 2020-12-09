# 项目背景

本项目数据来源于NLPCC会议的Open Domain Question Answering比赛。

在进行本项目之前，仔细研读了该比赛中冠军队伍的模型，并以该架构为baseline，尝试使用不同的模型进行优化。

本项目实战的文档总结：https://shimo.im/docs/TJ8KPj6JWrp6htjc/ 《KBQA项目实战》

# 任务描述

​	该任务的目标是基于给定KB(knowledge base)，构建自然语言问答系统(KBQA)。对于每个问题，从KBQA中选择相应实体，作为答案。

​	给定KB为三元组形式，即：subkect-predicate-object

​	根据问句中生成问题答案，需要分三个步骤：识别问句主体（subject）识别问句对应的谓词（predicate）查询知识库，得到（object），即问题答案。

​	我们分别从这三个步骤出发，去提升模型性能。

# 结果对比

baseline模型问句答案的F1 Score为：
​        ![img](https://uploader.shimo.im/f/sDwJKChzCEbVCLN1.png!thumbnail)
​      

本文模型问句答案的F1 Score为：
​        ![img](https://uploader.shimo.im/f/WblICGk2V0gQR9rW.png!thumbnail)
​      

# 代码运行

1，将知识图谱数据放到 ./data/下，数据地址：<http://tcci.ccf.org.cn/conference/2017/taskdata.php>

2，将训练好的bert模型参数放到 ./trained_bert_model/下

3，运行preprocessing.py

4，运行main.py生成答案

5，运行calF1生成F1score