# 根据模式匹配和baseline共同生成answer

import sys
import codecs
import time
import json
from scipy.spatial.distance import cosine
import code


import json
import torch
import re
import numpy as np
from model.modeling_albert import AlbertConfig, AlbertForSequenceClassification
from model import tokenization_albert
from model.file_utils import WEIGHTS_NAME

tokenizer = tokenization_albert.FullTokenizer(vocab_file='./trained_bert_model/albert_small/vocab.txt')
config = AlbertConfig.from_pretrained('./trained_bert_model/albert_small/',num_labels=2,finetuning_task='lcqmc')
model = AlbertForSequenceClassification.from_pretrained('./trained_bert_model/albert_small/',config=config)




space_pattern = re.compile(r'\s')
alphanum_pattern = re.compile(r'\(?[a-zA-Z0-9]+-?[a-zA-Z0-9]*\)?')
pattern_string = r'^你知道|请问|我想知道|谁知道|我很好奇|有谁知道|大家知道|有人知道|请告诉我|我想了解一下|请说出|告诉我|能告诉我|你了解|你清楚|' \
          r'你能说出|谁是|你能告诉我|我想请问|我想问问|我想问一下|我想了解|我很想知道|你们知道|问一下|我好奇|谁能告诉我|请问一下|你觉得|什么是'


def bert_output(question, predicate):
    question_sep = tokenizer.tokenize(question)
    predicate_sep = tokenizer.tokenize(predicate)
    text=["[CLS]"]+question_sep+["[SEP]"]+predicate_sep+["[SEP]"]
    indexed_tokens = tokenizer.convert_tokens_to_ids(text)
    token_type_ids = [0] * (len(question_sep) + 2) + [1] * (len(predicate_sep) + 1)
    tokens_tensor = torch.tensor([indexed_tokens])
    input = {'input_ids':tokens_tensor, 'token_type_ids': torch.tensor(token_type_ids)}
    model.eval()
    with torch.no_grad():
        outputs = model(**input)
        logits = outputs[:2][0]
        pred_ = logits.detach().cpu().numpy()
    return pred_[0][1]

def bert_model(kb, raw_question, correct_subject):
    subject, _ = space_pattern.subn('', correct_subject)
    SUB_question = raw_question.replace(subject, '(SUB)', 1)
    SUB_question = re.subn(r'\-', '', SUB_question)[0]
    rest_question = raw_question.replace(subject, '', 1)



    all_pre_list = []
    score_list = []

    all_pre_set = {pre for pre_answers in kb[correct_subject] for pre in pre_answers if len(pre) < len(SUB_question)}
    char_pre_set = {pre for pre in all_pre_set if alphanum_pattern.search(pre)}

    candidate_predicate = []

    if alphanum_pattern.search(rest_question):
        candidate_predicate = [pre for pre in char_pre_set if pre.lower() in SUB_question]
        if not candidate_predicate:
            candidate_predicate = [pre for pre in char_pre_set if alphanum_pattern.search(pre).group().lower() in SUB_question]

        if candidate_predicate:
            pre_length = [len(pre) for pre in candidate_predicate]
            return [candidate_predicate[np.argmax(pre_length)]]

    for predicate in all_pre_set:
        all_pre_list.append(predicate)
        pre, _ = alphanum_pattern.subn('', predicate)
        if pre:
            # predicate_pattern = SUB_question.strip() + '[SEP]' + pre
            pre_score = bert_output(SUB_question.strip(),  pre)
            score_list.append(pre_score)
        else:
            score_list.append(-float('inf'))

    sorted_list = sorted(zip(all_pre_list, score_list), key=lambda x: x[1], reverse=True)
    if len(sorted_list) > 1 and sorted_list[0][1] == sorted_list[1][1]:
        return [p[0] for p in sorted_list if p[1] == sorted_list[0][1]]
    if len(sorted_list) > 1 and sorted_list[0][1]-sorted_list[1][1]<0.2:
        return [sorted_list[0][0], sorted_list[1][0]]
    # if len(sorted_list) > 1 and sorted_list[1][1] > 0:
    #     if ('什么时候' in raw_question and sorted_list[0][1]-sorted_list[1][1]<1) or sorted_list[0][1]-sorted_list[1][1]<0.5:
    #         return [sorted_list[0][0], sorted_list[1][0]]
    return [sorted_list[0][0]]

def load_json_file(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
         decoded_object = json.load(f)

    return decoded_object


class answerCandidate:
    def __init__(self, sub='', pre='', qRaw='', qType=0, score=0, kbDict=[], wS=1, wP=10, wAP=100):
        self.sub = sub  # subject
        self.pre = pre  # predicate
        self.qRaw = qRaw  # raw question
        self.qType = qType  # question type
        self.score = score  # 分数
        self.kbDict = kbDict  # kd dictionary
        self.origin = ''
        self.scoreDetail = [0, 0, 0, 0, 0]
        self.wS = wS  # subject的权重
        self.wP = wP  # oredicate的权重
        self.wAP = wAP  # answer pattern的权重
        self.scoreSub = 0
        self.scoreAP = 0
        self.scorePre = 0

    def calcScore(self, qtList, countCharDict, debug=False, includingObj=[], vectorDict={}):
        # 最重要的部分，计算该答案的分数
        lenSub = len(self.sub)
        scorePre = 0
        scoreAP = 0
        pre = self.pre
        q = self.qRaw
        subIndex = q.index(self.sub)
        qWithoutSub1 = q[:subIndex]  # subject左边的部分
        qWithoutSub2 = q[subIndex + lenSub:]  # subject右边的部分

        qWithoutSub = q.replace(self.sub, '')  # 去掉subject剩下的部分
        qtKey = (self.qRaw.replace(self.sub, '(SUB)', 1) + ' ||| ' + pre)  # 把subject换成(sub)然后加上predicate
        if qtKey in qtList:
            scoreAP = qtList[qtKey]  # 查看当前的问题有没有在知识库中出现过

        self.scoreAP = scoreAP

        qWithoutSubSet1 = set(qWithoutSub1)
        qWithoutSubSet2 = set(qWithoutSub2)
        qWithoutSubSet = set(qWithoutSub)
        preLowerSet = set(pre.lower())
        # code.interact(local=locals())

        # 找出predicate和问题前后两部分的最大intersection
        intersection1 = qWithoutSubSet1 & preLowerSet
        intersection2 = qWithoutSubSet2 & preLowerSet

        if len(intersection1) > len(intersection2):
            maxIntersection = intersection1
        else:
            maxIntersection = intersection2

        # 计算来自predicate的分数，采用最大overlap的character的倒数 1/(n+1)
        preFactor = 0
        for char in maxIntersection:
            if char in countCharDict:
                preFactor += 1 / (countCharDict[char] + 1)
            else:
                preFactor += 1

        if len(pre) != 0:
            scorePre = preFactor / len(qWithoutSubSet | preLowerSet)
        else:
            scorePre = 0

        if len(includingObj) != 0 and scorePre == 0:
            for objStr in includingObj:
                scorePreTmp = 0
                preLowerSet = set(objStr.lower())
                intersection1 = qWithoutSubSet1 & preLowerSet
                intersection2 = qWithoutSubSet2 & preLowerSet

                if len(intersection1) > len(intersection2):
                    maxIntersection = intersection1
                else:
                    maxIntersection = intersection2

                preFactor = 0
                for char in maxIntersection:
                    if char in countCharDict:
                        preFactor += 1 / (countCharDict[char] + 1)
                    else:
                        preFactor += 1

                scorePreTmp = preFactor / len(qWithoutSubSet | preLowerSet)
                if scorePreTmp > scorePre:
                    scorePre = scorePreTmp

        if len(vectorDict) != 0 and len(pre) != 0:
            scorePre = 0

            # 找出所有在predicate中出现过的单词的词向量
            segListPre = []
            lenPre = len(pre)
            lenPreSum = 0
            for i in range(lenPre):
                for j in range(lenPre):
                    if i + j < lenPre:
                        preWordTmp = pre[i:i + j + 1]
                        if preWordTmp in vectorDict:
                            segListPre.append(preWordTmp)
                            lenPreSum += len(preWordTmp)

            # 找出所有在question当中出现过的单词的词向量
            lenQNS = len(qWithoutSub)
            segListQNS = []
            for i in range(lenQNS):
                for j in range(lenQNS):
                    if i + j < lenQNS:
                        QNSWordTmp = qWithoutSub[i:i + j + 1]
                        if QNSWordTmp in vectorDict:
                            segListQNS.append(QNSWordTmp)

            # Add Question type rules, ref to Table.1 in the article
            if qWithoutSub.find('什么时候') != -1 or qWithoutSub.find('何时') != -1:
                segListQNS.append('日期')
                segListQNS.append('时间')
            if qWithoutSub.find('在哪') != -1:
                segListQNS.append('地点')
                segListQNS.append('位置')
            if qWithoutSub.find('多少钱') != -1:
                segListQNS.append('价格')

            # 计算predicate和question之间的词向量cosine similarity
            for preWord in segListPre:
                scoreMaxCosine = 0
                for QNSWord in segListQNS:
                    # cosineTmp = lcs.cosine(vectorDict[preWord],vectorDict[QNSWord])
                    # cosineTmp = 1 - scipy.spatial.distance.cosine(vectorDict[preWord],vectorDict[QNSWord])
                    cosineTmp = 1 - cosine(vectorDict[preWord], vectorDict[QNSWord])
                    if cosineTmp > scoreMaxCosine:
                        scoreMaxCosine = cosineTmp
                scorePre += scoreMaxCosine * len(preWord)

            if lenPreSum == 0:
                scorePre = 0
            else:
                scorePre = scorePre / lenPreSum

            self.scorePre = scorePre

        scoreSub = 0

        # 计算subject的权重有多高，可能有些subject本身就是更重要一些，一般来说越罕见的entity重要性越高
        for char in self.sub:
            if char in countCharDict:
                scoreSub += 1 / (countCharDict[char] + 1)
            else:
                scoreSub += 1

        self.scoreSub = scoreSub
        self.scorePre = scorePre

        self.score = scoreSub * self.wS + scorePre * self.wP + scoreAP * self.wAP

        return self.score


def getAnswer(sub, pre, kbDict):
    answerList = []
    # kbDict[entityStr][len(kbDict[entityStr]) - 1][relationStr] = objectStr
    # 每个subject都有一系列的KB tiples，然后我们找出所有的subject, predicate, object triples
    for kb in kbDict[sub]:
        if pre in kb:
            answerList.append(kb[pre])

    return answerList


def base_model_subject(raw_question, kb, kb_set, vector, answer_pattern, count_char):

    kbDict = kb
    qtList = answer_pattern
    countCharDict = count_char
    vectorDict = vector
    wP = 10
    threshold = 0
    debug = False

    q = raw_question.strip().lower()
    candidateSet = set()
    result = ''
    maxScore = 0
    bestAnswer = set()


    candidate_subject_set = set()

    for i in range(1, len(q)):
        for j in range(len(q)-i+1):
            if q[j:i+j] in kb_set:
                candidate_subject_set.add(q[j:i+j])

    for sub in candidate_subject_set:
        for pos in kb[sub]:
            for pre in pos:
                newAnswerCandidate = answerCandidate(sub, pre, q, wP=10)
                candidateSet.add(newAnswerCandidate)

    candidateSetCopy = candidateSet.copy()
    if debug:
        print('len(candidateSet) = ' + str(len(candidateSetCopy)), end='\r', flush=True)
    candidateSet = set()

    candidateSetIndex = set()

    for aCandidate in candidateSetCopy:
        strTmp = str(aCandidate.sub + '|' + aCandidate.pre)
        if strTmp not in candidateSetIndex:
            candidateSetIndex.add(strTmp)
            candidateSet.add(aCandidate)

    # 针对每一个candidate answer，计算该candidate的分数，然后选择分数最高的作为答案
    for aCandidate in candidateSet:
        scoreTmp = aCandidate.calcScore(qtList, countCharDict, debug)
        if scoreTmp > maxScore:
            maxScore = scoreTmp
            bestAnswer = set()
        if scoreTmp == maxScore:
            bestAnswer.add(aCandidate)

    # 去除一些重复的答案
    bestAnswerCopy = bestAnswer.copy()
    bestAnswer = set()
    for aCandidate in bestAnswerCopy:
        aCfound = 0
        for aC in bestAnswer:
            if aC.pre == aCandidate.pre and aC.sub == aCandidate.sub:
                aCfound = 1
                break
        if aCfound == 0:
            bestAnswer.add(aCandidate)

    # 加入object的分数
    bestAnswerCopy = bestAnswer.copy()
    for aCandidate in bestAnswerCopy:
        if aCandidate.score == aCandidate.scoreSub:
            scoreReCal = aCandidate.calcScore(qtList, countCharDict, debug,
                                              includingObj=getAnswer(aCandidate.sub, aCandidate.pre, kbDict))
            if scoreReCal > maxScore:
                bestAnswer = set()
                maxScore = scoreReCal
            if scoreReCal == maxScore:
                bestAnswer.add(aCandidate)

    # 加入cosine similarity
    bestAnswerCopy = bestAnswer.copy()
    if len(bestAnswer) > 1:  # use word vector to remove duplicated answer
        for aCandidate in bestAnswerCopy:
            scoreReCal = aCandidate.calcScore(qtList, countCharDict, debug,
                                              includingObj=getAnswer(aCandidate.sub, aCandidate.pre, kbDict),
                                              vectorDict=vectorDict)
            if scoreReCal > maxScore:
                bestAnswer = set()
                maxScore = scoreReCal
            if scoreReCal == maxScore:
                bestAnswer.add(aCandidate)

    if debug:
        for ai in bestAnswer:
            for kb in kbDict[ai.sub]:
                if ai.pre in kb:
                    print(ai.sub + ' ' + ai.pre + ' ' + kb[ai.pre])
        return [bestAnswer, candidateSet]
    else:
        return bestAnswer



def answer_questions(question_file_path, answer_file_path):

    kb = load_json_file('kbJson.cleanPre.alias.utf8')
    vector = load_json_file('vectorJson.utf8')
    answer_pattern = load_json_file('outputAP')
    count_char = load_json_file('countChar')

    kb_set = set(kb)
    special_kb_set = set([sub for sub in kb_set if re.search(r'属于|现在|在|这', sub)])

    with open(question_file_path, 'r', encoding='utf-8') as f:
        standard_lines = f.readlines()

    question_num = 0
    start_time = time.time()
    with open(answer_file_path, 'w', encoding='utf-8') as fa:
        for i in range(len(standard_lines)):
            if standard_lines[i][:9] == '<question':
                fa.write(standard_lines[i])
                question_num += 1

            if standard_lines[i][:7] == '<triple':

                raw_question = standard_lines[i - 1][standard_lines[i - 1].find('\t') + 1:].strip().lower()
                raw_question, _ = space_pattern.subn('', raw_question)

                if re.search(pattern_string, raw_question):
                    question = re.sub(pattern_string, '', raw_question)
                else:
                    question = raw_question

                subject = ''

                if question in kb_set:
                    subject = question

                if subject == '' and re.search(r'[的|是]', question):
                    index1 = question.find('的') if question.find('的') != -1 else len(question)
                    index2 = question.find('是') if question.find('是') != -1 else len(question)
                    start_index = min(index1, index2)
                    if start_index != 0:
                        c = ''
                        for j in range(start_index, len(question)):
                            q1 = question[:j]
                            if q1 in kb_set:
                                c = q1
                            elif q1.upper() in kb_set:
                                c = q1.upper()
                        if c:
                            subject = c

                if subject == '' and re.search(r'属于|现在|在|这', question):
                    start_index = re.search(r'属于|现在|在|这', question).span()[0]
                    end_index = re.search(r'属于|现在|在|这', question).span()[1]
                    if start_index != 0:
                        c = ''
                        for j in range(end_index + 1, len(question)):
                            if question[:j] in special_kb_set:
                                c = question[:j]
                        q1 = question[:start_index]
                        if not c and q1 != '' and q1 in kb_set:
                            c = q1
                        if c:
                            subject = c

                if subject == '':
                    # b += 1
                    base_subject = base_model_subject(raw_question, kb, kb_set, vector, answer_pattern, count_char)
                    subject_candidate = [sub.sub for sub in base_subject]
                    sorted_subject_candidate = sorted(subject_candidate, key=lambda x: raw_question.find(x))
                    subject = sorted_subject_candidate[0]
                    # todo



                if subject.strip() not in kb_set:
                    bert_predicate = ''
                    fa.write(base_model_prediction[i])
                    fa.write(base_model_prediction[i + 1])
                    fa.write(base_model_prediction[i + 2])
                    continue
                else:
                    bert_predicate = bert_model(kb, raw_question, subject.strip())


                answers = []
                for pairs in kb[subject]:
                    for predicate in bert_predicate:
                        if predicate in pairs:
                            if pairs[predicate] not in answers:
                                answers.append(pairs[predicate])

                fa.write(standard_lines[i][:standard_lines[i].find('\t') + 1])
                for predicate in bert_predicate:
                    fa.write(subject + ' ||| ' + predicate.lower() + ' ||| ' + str(answers) + ' ====== ')
                fa.write('\n')
                answer_head = standard_lines[i + 1][:standard_lines[i + 1].find('\t') + 1]
                fa.write(answer_head)
                if len(answers) > 1:
                    for answer in answers:
                        fa.write(answer + ' | ')
                else:
                    fa.write(answers[0])
                fa.write('\n==================================================\n')

                print(
                    'processing ' + str(question_num) + 'th Q.\tAv time cost: ' + str((time.time() - start_time) / question_num)[:6] + ' sec')


if __name__ == '__main__':
    question_file_path = './data/nlpcc-iccpol-2016.kbqa.testing-data'
    answer_file_path = 'answer_albert'
    answer_questions(question_file_path, answer_file_path)