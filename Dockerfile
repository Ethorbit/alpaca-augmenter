FROM python:3.9 AS base
ARG UID=1000
ARG GID=1000
RUN groupadd -g "${GID}" python &&\
    useradd -m -g python -u "${UID}" python
USER python
WORKDIR /home/python

FROM base AS project_requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

FROM project_requirements AS nltk_installation
RUN python -m nltk.downloader all

FROM nltk_installation AS alpaca-agumenter
COPY ./src ./src
