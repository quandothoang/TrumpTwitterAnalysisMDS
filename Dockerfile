FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

COPY conda-linux-64.lock /tmp/conda-linux-64.lock

RUN conda install --quiet --yes make --file /tmp/conda-linux-64.lock \
    && conda clean --all -y -f \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "/home/${NB_USER}"

RUN python -m pip install deepchecks==0.18.1

# Install TinyTeX for PDF rendering
USER root
RUN wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh \
    && /root/.TinyTeX/bin/x86_64-linux/tlmgr install lmodern \
    && mv /root/.TinyTeX /opt/TinyTeX \
    && ln -s /opt/TinyTeX/bin/x86_64-linux/* /usr/local/bin/ \
    && fix-permissions /opt/TinyTeX
USER ${NB_USER}
