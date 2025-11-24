# Use Jupyter minimal-notebook as base image
FROM quay.io/jupyter/minimal-notebook:afe30f0c9ad8

# Switch to root to install packages
USER root

# Copy Linux conda lock file to container
COPY conda-linux-aarch64.lock /tmp/conda-linux-aarch64.lock

# Create conda environment using conda-lock
RUN mamba create --yes --name myenv --file /tmp/conda-linux-aarch64.lock

# Activate environment by default
ENV CONDA_DEFAULT_ENV=myenv
ENV PATH=/opt/conda/envs/myenv/bin:$PATH

# Switch back to notebook user
USER $NB_UID


