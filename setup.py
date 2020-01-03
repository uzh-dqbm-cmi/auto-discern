from setuptools import setup

setup(name='autodiscern',
      version='0.0.2',
      description='',
      url='https://github.com/CMI-UZH/auto-discern',
      packages=['autodiscern', 'autodiscern.experiments', 'autodiscern.predictors'],
      package_data={'autodiscern': ['package_data/*']},
      python_requires='>3.5.0',
      install_requires=[
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
            'pyyaml',
      ],
      extras_requires={
            'dev': ['jupyter', 'sacred', 'matplotlib'],
      },
      zip_safe=False)
