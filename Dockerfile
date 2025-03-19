# Use a lightweight Python image as the base
FROM python:3.10-slim

# Set a working directory inside the container
WORKDIR /home/kedro

# Install OS dependencies needed for Kedro or Git operations
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Install Kedro and Jupyter
RUN pip install kedro==0.18.10 jupyter

# By default, start a bash session
CMD ["/bin/bash"]