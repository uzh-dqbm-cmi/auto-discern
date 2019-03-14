from setuptools import setup

setup(name='autodiscern',
      version='0.0.1',
      description='',
      url='https://github.com/CMI-UZH/auto-discern',
      packages=['autodiscern'],
      install_requires=[
            'beautifulsoup4',
            'flake8',
            'pandas',
            'spacy',
      ],
      zip_safe=False)
