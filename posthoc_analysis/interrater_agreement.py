
import pandas as pd
import numpy as np
from autodiscern import DataManager


dm = DataManager('~/switchdrive/Institution/discern')
d = dm.load_transformed_data('2019-04-24_16-29-43_9dbdc11_doc_level')

questions = [4, 5, 9, 10, 11]
# questions = [5]
rater_ids = [5, 6]

agreement_scores = []


def classify(score):
    if score >= 3:
        return 1
    return 0


diffs = []

for q in questions:
    cnt = 0
    agree = 0
    for article_id in d:
        responses = d[article_id]['responses'].loc[q]
        if classify(responses.loc[rater_ids[0]]) == classify(responses.loc[rater_ids[1]]):
            agree += 1
        # else:
            # print('{}: {}, {}'.format(article_id,
            #                           responses.loc[rater_ids[0]],
            #                           responses.loc[rater_ids[1]]))
        diffs.append({'article_id': article_id,
                      'rater_5': responses.loc[rater_ids[0]],
                      'rater_6': responses.loc[rater_ids[1]],
                      })
        cnt += 1
    agreement = agree/cnt
    agreement_scores.append(agreement)
    print('Q {} Agreement: {:.0%}'.format(q, agreement))

print('Average agreement: {:.0%}'.format(sum(agreement_scores)/len(agreement_scores)))

# Q 4 Agreement: 91%
# Q 5 Agreement: 75%
# Q 9 Agreement: 83%
# Q 10 Agreement: 90%
# Q 11 Agreement: 94%
# Average agreement: 87%

# ground_truth = classify(np.round(responses.mean()))

df = pd.DataFrame(diffs)
pd.pivot_table(df, index='rater_5', columns='rater_6', values='article_id', aggfunc='count')

# rater_6  1     2    3
# rater_5
# 2        0     0     7
# 3        1    48     0
# 4        1    10     0

# rater_6   1 |  2 |  3 |  4 | 5  |
# rater_5+------------------------+
# 1      |(77)|  3 |  0 |  0 | 0  |
# -------+----+----+----+----+----+
# 2      |  4 |(16)|__7_|  0 | 0  |
# -------+----+----+----+----+----+
# 3      |__1_|_48_|(39)| 14 | 6  |
# -------+----+----+----+----+----+
# 4      |__1_|_10_| 23 |( 7)| 6  |
# -------+----+----+----+----+----+
# 5      |  0 |  0 |  2 |  3 |( 2)|
# -------+----+----+----+----+----+


# === compare both raters to the mean "ground truth"
agreement_scores = {}
for q in questions:
    agreement_scores[q] = {}
    for rater_id in rater_ids:
        agreement_scores[q][rater_id] = []


def list_mean(l):
    return sum(l)/len(l)


for q in questions:
    for article_id in d:
        responses = d[article_id]['responses'].loc[q]
        ground_truth = classify(int(np.round(responses.mean())))
        print(responses)
        print(ground_truth)
        for rater_id in agreement_scores[q]:
            agrees_with_ground_truth = ground_truth == classify(responses.loc[rater_id])
            print(agrees_with_ground_truth)
            agreement_scores[q][rater_id].append(agrees_with_ground_truth)
        print()


rater_agreement_scores = {q: [] for q in questions}
for q in questions:
    print("=== Question {} ===".format(q))
    for rater_id in agreement_scores[q]:
        score = list_mean(agreement_scores[q][rater_id])
        rater_agreement_scores[q].append(score)
        print('Rater {}: {:.0%}'.format(rater_id, score))

    print('Average agreement: {:.0%}'.format(list_mean(rater_agreement_scores[q])))

# === Question 4 ===
# Rater 5: 94%
# Rater 6: 97%
# Average agreement: 96%
# === Question 5 ===
# Rater 5: 81%
# Rater 6: 94%
# Average agreement: 88%
# === Question 9 ===
# Rater 5: 90%
# Rater 6: 93%
# Average agreement: 92%
# === Question 10 ===
# Rater 5: 96%
# Rater 6: 94%
# Average agreement: 95%
# === Question 11 ===
# Rater 5: 99%
# Rater 6: 95%
# Average agreement: 97%
