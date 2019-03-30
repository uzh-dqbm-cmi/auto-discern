import os
import glob
import pandas as pd
from pathlib import Path


class DataManager:
    """Data manager that loads DISCERN corpus into different formats.
    
       Args:
           data_pth: string, path/to/data
                        
       .. note :: 

           On windows OS, the backslash in the string should be escaped!!
    """

    def __init__(self, data_path):
        expanded_path = Path(data_path).expanduser()
        if expanded_path.exists():
            self.data_path = expanded_path
        else:
            raise ValueError("Path does not exist: {}".format(data_path))
        self.data = {}

    def _load_articles(self, version_id: str) -> None:
        """Loads all files located in self.data_path/data/<version_id> into a pd.df at self.data dict[version_id]"""

        print("Loading articles...")
        target_ids = pd.read_csv(Path(self.data_path, "data/target_ids.csv"))

        articles = pd.DataFrame(columns=['entity_id', 'content'])
        articles_path = Path(self.data_path, "data/{}/*".format(version_id))

        files = glob.glob(articles_path.__str__())

        for file in files:
            # to enforce encoding -- it generates errors without it!
            with open(file, 'r', encoding='utf-8') as f:
                # keep what's after the last / and before the .
                print("file:", file)
                entity_id = os.path.basename(file).split('.')[0]
                print("entity_id:", entity_id)
                content = f.read()
                articles = articles.append({'entity_id': int(entity_id), 'content': content}, ignore_index=True)
        else:
            print("WARNING: no files found at {}".format(articles_path))
        articles['entity_id'] = articles['entity_id'].astype(int)
        articles = pd.merge(articles, target_ids, on='entity_id')

        print("{} articles loaded".format(articles.shape[0]))
        self.data[version_id] = articles

    def _load_responses(self) -> None:
        print("Loading responses...")
        self.data['responses'] = pd.read_csv(Path(self.data_path, "data/responses.csv"))
        print("{} responses loaded".format(self.data['responses'].shape[0]))

    def _articles(self, version) -> pd.DataFrame:
        version_id = "{}_articles".format(version)
        if version_id not in self.data:
            self._load_articles(version_id)
        return self.data[version_id]

    @property
    def html_articles(self) -> pd.DataFrame:
        return self._articles('html')

    @property
    def clean_articles(self) -> pd.DataFrame:
        return self._articles('cleaned_text')

    @property
    def selected_html_articles(self) -> pd.DataFrame:
        return self._articles('remove_selected_html')

    @property
    def no_html_articles(self) -> pd.DataFrame:
        return self._articles('remove_all_html')

    @property
    def responses(self) -> pd.DataFrame:
        if 'responses' not in self.data:
            self._load_responses()
        return self.data['responses']
