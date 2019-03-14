from setuptools import setup

setup(name='autodiscern',
      version='0.0.1',
      description='',
      url='https://github.com/CMI-UZH/auto-discern',
      packages=['autodiscern'],
      python_requires='>3.5.0',
      install_requires=[
            'allennlp',
            'beautifulsoup4',
            'flake8',
            'pandas',
            'spacy',
      ],
      zip_safe=False)
