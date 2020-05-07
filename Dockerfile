# Use an official Python runtime as a parent image
# tried to use a smaller python:3.6-slim image, but wheels won't build
FROM python:3.6 as pythonbuilder

COPY autodiscern /autodiscern/autodiscern
COPY setup.py /autodiscern
WORKDIR /autodiscern

# Create a virtual env and install dependencies and the project package itself
RUN  python3 -m venv /venv \
  && /venv/bin/pip install --trusted-host pypi.python.org . \
  && /venv/bin/python3 -c "import nltk; nltk.download('vader_lexicon')" \
  && /venv/bin/python3 -m spacy download en_core_web_sm

# ================================================================
FROM rappdw/docker-java-python as metamapbuilder

## copy over the virtual env from the pythonbuilder image
COPY --from=pythonbuilder /venv /venv
COPY --from=pythonbuilder /root/nltk_data /root/nltk_data

# copy over the data files, including the metamap executable
COPY data /data
COPY validator_site /app

# Install pymetamap from git separately
#RUN git clone https://github.com/orisenbazuru/pymetamap.git \
RUN git clone https://github.com/AnthonyMRios/pymetamap.git \
 && cd pymetamap && /venv/bin/python3 setup.py install

WORKDIR /app

# Make port 80 available to the world outside this container
EXPOSE 80

# this image doesn't have git installed, so silence errors (we don't need git anyway)
# may be outdated as of docker-java-python image
ENV GIT_PYTHON_REFRESH quiet

# Run app.py when the container launches
CMD ["/venv/bin/python3", "app.py"]