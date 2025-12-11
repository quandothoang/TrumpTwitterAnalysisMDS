FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN conda install --quiet --yes make --file /tmp/conda-linux-64.lock \
    && conda clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

RUN python -m pip install deepchecks==0.18.1

# Install TinyTeX v2023.12 (pinned version for reproducibility)
RUN wget -qO- "https://github.com/rstudio/tinytex-releases/releases/download/v2023.12/TinyTeX-1-v2023.12.tar.gz" \
    | tar -xz -C /home/${NB_USER} \
    && /home/${NB_USER}/.TinyTeX/bin/x86_64-linux/tlmgr install lmodern \
    && fix-permissions "/home/${NB_USER}"

ENV PATH="/home/${NB_USER}/.TinyTeX/bin/x86_64-linux:${PATH}"