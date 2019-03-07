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
    └── html/
        └── *.html

```

### Example Usage

```python
# IPython magics for auto-reloading code changes to the library
%load_ext autoreload
%autoreload 2

import autodiscern as ad

# see "Note on Data" above for what to pass here
dm = ad.DataManager("path/to/discern/data")

# data is loaded in automatically
dm.articles.head()
dm.responses.head()

```