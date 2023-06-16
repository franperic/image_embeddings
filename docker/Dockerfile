# Base Image
FROM python:3.9.6

COPY requirements.txt /home/.

# Pin pip version
RUN python -m pip install --upgrade pip

# Install packages from requirements
RUN pip install -r /home/requirements.txt

RUN pip install git+https://github.com/rwightman/pytorch-image-models.git 

EXPOSE 8080
COPY . ./app

WORKDIR /app

# Set the entrypoint to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8080", "--allow-root", "--no-browser"]
