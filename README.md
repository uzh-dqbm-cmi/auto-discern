# auto-discern

Automating the application of the [DISCERN](http://www.discern.org.uk/index.php) instrument to rate the quality of health information on the web. 

### How to Use this Repo
* `git clone` the repo and `cd` into it.
* Run `pip install -e .` to install the repo's python package.
* Acquire a copy of this project's data and structure it according to "A Note on Data" below. 
* Skip on down to Example Usage below.

### A Note on Data
This repo contains no data. To use this package, you must have a copy of the data locally, in the following file structure:

```
path/to/discern/data/
└── data/
    ├── target_ids.csv
    ├── responses.csv
    └── html_articles/
        └── *.html

```

### Notebooks

Please follow this notebook naming convention for exploratory notebooks in the shared Switchdrive folder: 
`<number>_<initials>_<short_description>.ipynb`. 

### Example Usage

```python
# IPython magics for auto-reloading code changes to the library
%load_ext autoreload
%autoreload 2

import autodiscern as ad
import autodiscern.transformations as adt

# see "Note on Data" above for what to pass here
dm = ad.DataManager("path/to/discern/data")

# data is loaded in automatically
dm.html_articles.head()
dm.responses.head()

input_dicts = [{'id': row['entity_id'], 'content': row['content']} 
              for i, row in dm.html_articles.iterrows()]

# select which transformation you want to apply
transforms = [
    # adt.remove_html,
    adt.remove_html_to_sentences,
    # adt.remove_selected_html,
]
transformer = adt.Transformer(transforms, num_cores=4)

transformed_data = transformer.apply(input_dicts)

for item in transformed_data[:5]:
    print(item)

# ===
# tag Named Entities
from allennlp.predictors.predictor import Predictor
from IPython.display import HTML
ner_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
ner = []
# look at the first 50 sentences of the first document
for sentence in transformed_data[0]['content'][:50]:
    ner.append(adt.allennlp_ner_tagger(sentence, ner_predictor))
HTML(adt.ner_tuples_to_html(ner))

```

## MetaMap

### Setup Instructions for MetaMap
* Download MetaMapLite:
    * Download MetaMapLite from [here](https://metamap.nlm.nih.gov/MetaMapLite.shtml). You will need to request a license to access the download, which takes a few hours.
    * Place the zip file in a new directory called `metamap`, and unzip.
    * If necessary, install Java as per metamap instructions.
    * Test metamap by creating a test.txt file with the contents "John had a huge heart attack". Run `./metamap.sh test.txt`. A new file, test.mmi, shoudl be created with details about the Myocardial Infarction concept.
* Install `pymetamap` wrapper:
    * (A working version of `pymetamap` compatible with MetaMapLite is on someone's forked repo's branch)
    * `git clone https://github.com/kaushikacharya/pymetamap.git`
    * `git checkout lite`
    * Inside your project environment: `python setup.py install`

### To use `pymetamap`
`pymetamap` ingests text and returns `NamedTuples` for each MetaMap concept with named fields.
```python
from pymetamap import MetaMapLite
# insert the path to your parent `metamap` dir here
mm = MetaMapLite.get_instance('/Users/laurakinkead/Documents/metamap/public_mm_lite/')

sents = ['Heart Attack', 'John had a huge heart attack']
concepts, error = mm.extract_concepts(sents,[1,2])

for concept in concepts:
    for fld in concept._fields:
        print("{}: {}".format(fld, getattr(concept, fld)))
    print("\n")
```
prints:
```python
index: 2
mm: MMI
score: 3.75
preferred_name: Myocardial Infarction
cui: C0027051
semtypes: [dsyn]
trigger: "Heart Attack"-text-0-"heart attack"-NN-0
pos_info: 17/12
tree_codes: C14.280.647.500;C14.907.585.500
```
