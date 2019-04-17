import datetime
import glob
import os
import pandas as pd
import pickle
from pathlib import Path

import autodiscern.transformations as adt


class DataManager:
    """Data manager that loads DISCERN corpus into different formats.

       Args:
           data_path: string, path/to/data

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

    def build_dicts(self):
        """Build a dictionary of data dictionaries, keyed on their entity_ids. """
        self._load_articles('html_articles')
        self._load_responses()
        print("Building data dicts...")
        data_list_of_dicts = self.data['html_articles'].to_dict('records')
        data_dict = adt.convert_list_of_dicts_to_dict_of_dicts(data_list_of_dicts)

        for id in data_dict:
            entity_responses = self.data['responses'][self.data['responses']['entity_id'] == id]
            responses_pivoted = pd.pivot_table(entity_responses, index='questionID', columns='uid', values='answer',
                                               aggfunc='median')
            data_dict[id]['responses'] = responses_pivoted

        ids = list(data_dict.keys())
        print(" ... {} data dicts built".format(len(ids)))
        print("Available keys: {}".format(", ".join(list(data_dict[ids[0]].keys()))))
        return data_dict

    def _load_articles(self, version_id: str) -> None:
        """Loads all files located in self.data_path/data/<version_id> into a pd.df at self.data dict[version_id]"""

        print("Loading articles...")

        articles = pd.DataFrame(columns=['entity_id', 'content'])
        articles_path = Path(self.data_path, "data/{}/*".format(version_id))

        files = glob.glob(articles_path.__str__())

        if len(files) == 0:
            print("WARNING: no files found at {}".format(articles_path))

        for file in files:
            # to enforce encoding -- it generates errors without it!
            with open(file, 'r', encoding='utf-8') as f:
                # keep what's after the last / and before the .
                entity_id = os.path.basename(file).split('.')[0]
                content = f.read()
                articles = articles.append({'entity_id': int(entity_id), 'content': content}, ignore_index=True)

        # only keep articles listed in the target_ids file
        target_ids = pd.read_csv(Path(self.data_path, "data/target_ids.csv"))
        articles['entity_id'] = articles['entity_id'].astype(int)
        articles = pd.merge(articles, target_ids, on='entity_id')

        # add the article urls
        article_urls = pd.read_csv(Path(self.data_path, "data/urls.csv"))
        articles = pd.merge(articles, article_urls, on='entity_id')

        print(" ... {} articles loaded".format(articles.shape[0]))
        self.data[version_id] = articles

    def _load_responses(self) -> None:
        print("Loading responses...")
        self.data['responses'] = pd.read_csv(Path(self.data_path, "data/responses.csv"))
        print(" ... {} responses loaded".format(self.data['responses'].shape[0]))

    def _articles(self, version) -> pd.DataFrame:
        version_id = "{}_articles".format(version)
        if version_id not in self.data:
            self._load_articles(version_id)
        return self.data[version_id]

    def save_transformed_data(self, data, tag=None):
        """Save a data dictionary to data/transformed directory with a filename created from the current timestamp and
        an optional tag. """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if tag:
            filepath = Path(self.data_path, "data/transformed_data/{}_{}.pkl".format(timestamp, tag))
        else:
            filepath = Path(self.data_path, "data/transformed_data/{}.pkl".format(timestamp))
        with open(filepath, "wb+") as f:
            pickle.dump(data, f)

    def load_transformed_data(self, filename):
        """Load a pickled data dictionary from the data/transformed directory.
        filename can be provided with or without the .pkl extension"""

        if '.pkl' in filename:
            filepath = Path(self.data_path, "data/transformed_data/{}".format(filename))
        else:
            filepath = Path(self.data_path, "data/transformed_data/{}.pkl".format(filename))
        with open(filepath, "rb+") as f:
            return pickle.load(f)

    def load_most_recent_transformed_data(self):
        """Load the most recent pickled data dictionary from the data/transformed directory,
        as determined by the timestamp in the filename. """

        filepath = Path(self.data_path, "data/transformed_data/*")
        files = glob.glob(filepath.__str__())

        if len(files) == 0:
            print("ERROR: no files found at {}".format(filepath))
            return

        filenames = [os.path.basename(file) for file in files]
        filenames.sort()
        chosen_one = filenames[-1]
        print("Loading {}".format(chosen_one))
        return self.load_transformed_data(chosen_one)

    @property
    def html_articles(self) -> pd.DataFrame:
        return self._articles('html')

    @property
    def clean_articles(self) -> pd.DataFrame:
        print("WARNING: this function is deprecated")
        return self._articles('cleaned_text')

    @property
    def selected_html_articles(self) -> pd.DataFrame:
        print("WARNING: this function is deprecated")
        return self._articles('remove_selected_html')

    @property
    def no_html_articles(self) -> pd.DataFrame:
        print("WARNING: this function is deprecated")
        return self._articles('remove_all_html')

    @property
    def responses(self) -> pd.DataFrame:
        if 'responses' not in self.data:
            self._load_responses()
        return self.data['responses']

    def _load_metamap_semantics(self):
        self.data['metamap_semantics'] = pd.read_csv(Path(self.data_path, "data/metamap/metamap_semantics.csv"))

    @property
    def metamap_semantics(self) -> pd.DataFrame:
        if 'metamap_semantics' not in self.data:
            self._load_metamap_semantics()
        return self.data['metamap_semantics']
