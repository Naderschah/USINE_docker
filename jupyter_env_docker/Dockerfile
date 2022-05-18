# Set the base image using miniconda 
FROM jupyter/scipy-notebook:latest

# Set working directory 
WORKDIR /home/notebooks

# Add requirements file 
ADD requirements.txt /app/

# Installs, clean, and update   
RUN pip install -r /app/requirements.txt


#CMD jupyter notebook --port=8000 --no-browser --ip=0.0.0.0 --allow-root