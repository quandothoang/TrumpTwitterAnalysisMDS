FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN conda install --quiet --yes make --file /tmp/conda-linux-64.lock \
    && conda clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

RUN python -m pip install deepchecks==0.18.1

# Install LaTeX for PDF rendering
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    lmodern \
    texlive-latex-base \
    texlive-latex-recommended \
    texlive-latex-extra \
    texlive-luatex \
    texlive-fonts-recommended \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
USER ${NB_USER}
