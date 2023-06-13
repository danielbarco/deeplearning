FROM pytorch/pytorch:latest
ENV TZ=Europe/Zurich
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git  -y

RUN pip install scikit-learn ipykernel torchsummary matplotlib torchvision numpy scipy pandas tqdm
