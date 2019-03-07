import glob
import pandas as pd
from pathlib import Path


class DataManager:

    def __init__(self, data_path):
        expanded_path = Path(data_path).expanduser()
        if expanded_path.exists():
            self.data_path = expanded_path
        else:
            raise ValueError("Path does not exist: {}".format(data_path))

        self.data = {}

    def _load_articles(self):
        print("Loading articles...")
        target_ids = pd.read_csv(Path(self.data_path, "data/target_ids.csv"))

        articles = pd.DataFrame(columns=['entity_id', 'html'])
        articles_path = Path(self.data_path, "data/html/*.html")

        files = glob.glob(articles_path.__str__())
        for file in files:
            with open(file, 'r') as f:
                entity_id = int(file.split('/')[-1].replace('.html', ''))
                content = f.read()
                articles = articles.append({'entity_id': int(entity_id), 'html': content}, ignore_index=True)

        articles['entity_id'] = articles['entity_id'].astype(int)
        articles = pd.merge(articles, target_ids, on='entity_id')

        print("{} articles loaded".format(articles.shape[0]))
        self.data['articles'] = articles

    def _load_responses(self):
        print("Loading responses...")
        self.data['responses'] = pd.read_csv(Path(self.data_path, "data/responses.csv"))
        print("{} responses loaded".format(self.data['responses'].shape[0]))

    @property
    def articles(self):
        if 'articles' not in self.data:
            self._load_articles()
        return self.data['articles']

    @property
    def responses(self):
        if 'responses' not in self.data:
            self._load_responses()
        return self.data['responses']
