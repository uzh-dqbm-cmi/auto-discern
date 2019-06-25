from setuptools import setup

setup(name='autodiscern',
      version='0.0.1',
      description='',
      url='https://github.com/CMI-UZH/auto-discern',
      packages=['autodiscern', 'autodiscern.experiments', 'autodiscern.predictors'],
      python_requires='>3.5.0',
      install_requires=[
            # 'allennlp==0.8.2',
            'beautifulsoup4',
            'flake8',
            'flask',
            'gitpython',
            'jsonnet==0.10.0',
            'nltk',
            'numpy>=1.15.0',
            'pandas==0.24.1',
            'scikit-learn',
            'spacy==2.0.18',
            'tldextract',
      ],
      extras_requires={
            'allennlp': ['allennlp'],
      },
      zip_safe=False)
