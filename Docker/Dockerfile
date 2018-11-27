FROM fedora:28
LABEL maintainer="Daehyun You <daehyun@dc.tohoku.ac.jp>"

# install packages using dnf
ENV LANG=en_US.UTF-8
RUN dnf update -y \
    && dnf install -y \
        which bzip2 git texlive nodejs \
        https://downloads.sourceforge.net/project/mscorefonts2/rpms/msttcore-fonts-installer-2.6-1.noarch.rpm \
    && dnf clean all

# install Anaconda
WORKDIR /root
ENV PATH /opt/anaconda/bin:$PATH
ADD https://repo.anaconda.com/archive/Anaconda3-5.2.0-Linux-x86_64.sh anaconda.sh
RUN bash anaconda.sh -bp /opt/anaconda \
    && rm anaconda.sh

# build Pipenv project
WORKDIR /app
ENV PIPENV_VENV_IN_PROJECT=true \
    PATH=/app/.venv/bin:$PATH \
    PYSPARK_PYTHON=/app/.venv/bin/python
COPY Pipfile /app/
RUN pip install pipenv \
    && pipenv --python /opt/anaconda/bin/python --site-packages \
    && pipenv install \
    && pipenv install --dev \
    && python -m jupyter labextension install \
        @jupyterlab/toc \
    && rm -fr /root/.cache
RUN chmod a+wx /app

EXPOSE 4040 8888
CMD [ "python" ]
