FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git  -y

RUN pip install scikit-learn ipykernel torchsummary matplotlib