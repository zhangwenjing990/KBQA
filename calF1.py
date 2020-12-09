import re
import torch.nn as nn

pattern_space = re.compile(r'\s')
subject_pattern = re.compile(r'《(.*)》')
predicate_pattern = re.compile(r'[·•\-\s]|(\[.*\]$)')
p = re.compile(r'\s')

with open('./data/nlpcc-iccpol-2016.kbqa.testing-data', 'r', encoding='utf-8') as f:
    standard_lines = f.readlines()

with open('answer_albert', 'r', encoding='utf-8') as f:
    prediction_lines = f.readlines()

F1_answer = 0
F1_predicate = 0
question_counts = 0
correct_subjects = 0
correct_predicates = 0
correct_answers = 0

for i in range(len(standard_lines)):

    if standard_lines[i][1:7] == 'triple':
        question_counts += 1

        standard_triple = standard_lines[i]
        standard_subject = standard_triple[standard_triple.find('\t') + 1:standard_triple.find(' ||| ')].strip().lower()
        standard_subject, _ = pattern_space.subn('', standard_subject)
        standard_subject = subject_pattern.sub(r'\1', standard_subject)
        standard_predicate_and_answer = standard_triple[standard_triple.find(' ||| ')+5:].strip()
        standard_predicate = standard_predicate_and_answer[:standard_predicate_and_answer.find(' ||| ')].strip().lower()
        standard_predicate, _ = predicate_pattern.subn('', standard_predicate)
        standard_predicate_set = {standard_predicate}


        prediction_triple = prediction_lines[i]
        prediction_subject = prediction_triple[prediction_triple.find('\t') + 1:prediction_triple.find(' ||| ')].strip().lower()
        prediction_subject, _ = pattern_space.subn('', prediction_subject)
        prediction_subject = subject_pattern.sub(r'\1', prediction_subject)
        prediction_predicates = prediction_triple

        predicate_num = re.findall(r' ====== ', prediction_predicates)
        prediction_predicates_set = set()
        for j in range(len(predicate_num)):
            prediction_predicates = prediction_predicates[prediction_predicates.find(' ||| ') + 5:]
            pre = prediction_predicates[:prediction_predicates.find(' ||| ')].strip().lower()
            pre, _ = predicate_pattern.subn('', pre)
            prediction_predicates_set.add(pre)
            prediction_predicates = prediction_predicates[prediction_predicates.find(' ====== ') + 8:]


        if standard_subject == prediction_subject:
            correct_subjects += 1

        intersection = len(standard_predicate_set & prediction_predicates_set)
        recall = intersection / len(standard_predicate_set)

        if intersection:
            F1_predicate += 2 * recall * precision / (recall + precision)

        else:

            # if standard_subject == prediction_subject:
            print('Question:   ' + standard_lines[i - 1][standard_lines[i - 1].find('\t') + 1:].strip())
            print('Standard:   ' + standard_subject +' ||| ' + ' | '.join(standard_predicate_set))
            print('Prediction: ' + prediction_subject + ' ||| ' + ' | '.join(prediction_predicates_set))


    if standard_lines[i][1:7] == 'answer':

        standard_answers = standard_lines[i][standard_lines[i].index('\t')+1:].strip().lower()
        standard_answer_set = set()
        index = standard_answers.find(' | ')
        if index == -1:
            standard_answers, n = p.subn('',standard_answers)
            standard_answer_set.add(standard_answers)
        while standard_answers.find(' | ') != -1:
            index = standard_answers.index(' | ')
            answer = standard_answers[:index]
            answer, n = p.subn('',answer)
            standard_answer_set.add(answer)
            standard_answers = standard_answers[index+3:]

        standard_answers = standard_answer_set


        prediction_answers = prediction_lines[i][prediction_lines[i].index('\t') + 1:].strip('\n').lower()
        prediction_answer_set = set()
        indexS = prediction_answers.find(' | ')
        if indexS == -1:
            answerCell, n = p.subn('', prediction_answers)
            prediction_answer_set.add(answerCell)
        while prediction_answers.find(' | ') != -1:
            indexS = prediction_answers.index(' | ')
            answer = prediction_answers[:indexS]
            answer, n = p.subn('', answer)
            prediction_answer_set.add(answer)
            prediction_answers = prediction_answers[indexS + 3:]

        prediction_answers = prediction_answer_set


        intersection = len(prediction_answers & standard_answers)
        recall = intersection / len(standard_answers)
        precision = intersection / len(prediction_answers) if len(prediction_answers) else 0

        if intersection:
            F1_answer += 2 * recall * precision / (recall + precision)
        #
        # else:
        #     print('Question:   ' + standard_lines[i - 2][standard_lines[i - 2].find('\t') + 1:].strip())
        #     print('Standard:   ' + standard_lines[i - 1][standard_lines[i - 1].find('\t') + 1:].strip())
        #     print('Prediction: ' + prediction_lines[i - 1][prediction_lines[i - 1].find('\t') + 1:].strip())

        # F1 += 2*len(prediction_answers & standard_answers)/(len(prediction_answers) + len(standard_answers))
        if len(prediction_answers & standard_answers) != 0:
            correct_answers += 1
        # else:
        #     print(prediction_answers)
        #     print(standard_answers)



# print(question_counts)
# print(correct_subjects)
# print('Predicate Accuracy of Questions:\033[36m'+str(correct_predicates/question_counts))
# print('Subject Accuracy of Questions:\033[36m'+str(correct_subjects/question_counts))
# print(correct_answers/question_counts)

print('\033[0mTriple F1:\t\033[32m' + str(F1_predicate/question_counts))
print('\033[0mAnswer F1:\t\033[32m' + str(F1_answer/question_counts))
