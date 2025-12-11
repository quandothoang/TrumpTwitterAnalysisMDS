FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN conda install --quiet --yes make --file /tmp/conda-linux-64.lock \
    && conda clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

RUN python -m pip install deepchecks==0.18.1

# Install TinyTeX for PDF rendering
RUN quarto install tinytex --no-prompt \
    && fix-permissions "/home/${NB_USER}"
