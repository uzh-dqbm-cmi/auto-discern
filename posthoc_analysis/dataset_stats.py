import pandas as pd
from datatc import DataManager


disease_category_map = {
    2: 'breast cancer',
    4: 'arthritis',
    5: 'depression',
}

dm = DataManager('/opt/data/autodiscern/')
# data_file = 'transformed_data/2019-05-02_15-49-09_a0745f9_sent_level_MM.pkl'
data = dm['transformed_data'].select('pkl').load()

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

for topic in data_stats:
    data_stats[topic]['Number of Articles'] = len(set(data_stats[topic]['Articles']))
    data_stats[topic].pop('Articles', None)

stats_df = pd.DataFrame(data_stats).T
stats_df['Avg Number of Sentences per Article'] = stats_df['Number of Sentences'] / stats_df['Number of Articles']
stats_df['Avg Number of Words per Article'] = stats_df['Number of Words'] / stats_df['Number of Articles']
stats_df.T

#                                      breast cancer      arthritis     depression
# Number of Articles                       79               88              102
# Number of Sentences                   10170            10950            13790
# Number of Words                      125891           129759           160597
# Avg Number of Sentences per Article     129              124              135
# Avg Number of Words per Article        1594             1475             1574
