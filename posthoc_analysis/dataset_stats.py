import pandas as pd
from datatc import DataManager


disease_category_map = {
    2: 'breast cancer',
    4: 'arthritis',
    5: 'depression',
}

dm = DataManager('/opt/data/autodiscern/data/transformed_data')
# data_file = 'transformed_data/2019-05-02_15-49-09_a0745f9_sent_level_MM.pkl'
data = dm.data_directory.select('pkl').load()

data_stats = {}
for topic_id in disease_category_map:
    topic = disease_category_map[topic_id]
    data_stats[topic] = {
        'Articles': [],
        'Number of Articles': 0,
        'Number of Sentences': 0,
        'Number of Words': 0,
    }

for id in data:
    topic_id = data[id]['categoryName']
    topic = disease_category_map[topic_id]
    data_stats[topic]['Articles'].append(data[id]['entity_id'])
    data_stats[topic]['Number of Sentences'] += 1
    data_stats[topic]['Number of Words'] += len(data[id]['content'].split(' '))

# gather scores for each doc (not each sentence!)
for q in [4, 5, 9, 10, 11]:
    for topic in data_stats:
        scores = []
        for doc_id in list(set(data_stats[topic]['Articles'])):
            doc_class = int(round(data['{}-0'.format(doc_id)]['responses'].loc[q].mean()) > 3)
            scores.append(doc_class)
        data_stats[topic]['Q{} % Positive Class'.format(q)] = '{:.0%}'.format(sum(scores)/len(scores))

for topic in data_stats:
    data_stats[topic]['Number of Articles'] = len(set(data_stats[topic]['Articles']))
    # data_stats[topic].pop('Articles', None)
    for q in [4, 5, 9, 10, 11]:
        data_stats[topic]['Q{} % Positive Class'.format(q)] = round(data[id]['responses'].loc[q].mean())

stats_df = pd.DataFrame(data_stats).T
stats_df['Avg Number of Sentences per Article'] = stats_df['Number of Sentences'] / stats_df['Number of Articles']
stats_df['Avg Number of Words per Article'] = stats_df['Number of Words'] / stats_df['Number of Articles']

columns = ['Number of Articles', 'Number of Sentences', 'Number of Words',
           'Avg Number of Sentences per Article', 'Avg Number of Words per Article',
           'Q4 % Positive Class', 'Q5 % Positive Class', 'Q9 % Positive Class', 'Q10 % Positive Class',
           'Q11 % Positive Class',
           ]

stats_df[columns].T

#                                      breast cancer      arthritis     depression
# Number of Articles                       79               88              102
# Number of Sentences                   10170            10950            13790
# Number of Tokens                      125891           129759           160597
# Avg Number of Sentences per Article     129              124              135
# Avg Number of Words per Article        1594             1475             1574
# Q4 Positive Class Prevelance            13%              14%              14%
# Q5 Positive Class Prevelance            20%              26%              24%
# Q9 Positive Class Prevelance            85%              28%              52%
# Q10 Positive Class Prevelance           89%              80%              65%
# Q11 Positive Class Prevelance           63%              16%              33%
