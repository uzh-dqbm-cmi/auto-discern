# Use an official Python runtime as a parent image
FROM python:3.6 as pythonbuilder

COPY autodiscern /autodiscern/autodiscern
COPY setup.py /autodiscern
WORKDIR /autodiscern

# Create a virtual env and install dependencies and the project package itself
RUN  python3 -m venv /venv \
  && /venv/bin/pip install --trusted-host pypi.python.org . \
  && /venv/bin/python3 -c "import nltk; nltk.download('vader_lexicon')"

# ================================================================
FROM rappdw/docker-java-python as metamapbuilder

COPY --from=pythonbuilder /venv /venv
COPY --from=pythonbuilder /root/nltk_data /root/nltk_data

# copy over the data files, including the metamap executable
COPY data /data
COPY validator_site /app

# Install pymetamap from git separately
#RUN git clone https://github.com/orisenbazuru/pymetamap.git \
RUN git clone https://github.com/AnthonyMRios/pymetamap.git \
 && cd pymetamap && /venv/bin/python3 setup.py install

## ================================================================
## use a smaller image for the one we will actually upload
#FROM python:3.6-slim
#
## copy over the virtual env from the builder image
#COPY --from=metamapbuilder /venv /venv
#COPY --from=metamapbuilder /metamap /metamap
#
## copy over only the files we need for going live
#COPY validator_site /app
#COPY data/metamap /data/metamap
#COPY data/models /data/models
WORKDIR /app

# Make port 80 available to the world outside this container
EXPOSE 80

# this image doesn't have git installed, so silence errors (we don't need git anyway)
ENV GIT_PYTHON_REFRESH quiet

# Run app.py when the container launches
CMD ["/venv/bin/python3", "app.py"]