# deeplearning
UZH deep learning FS23

Use conda to install an environment, then install dependencies from environment.yml
```bash
conda env create --name envname --file=environments.yml
```
How to start a docker

```bash
docker build . -t deeplearning
docker run --shm-size 64G -it -v /cluster/home/baoc/uzh/deeplearning:/workspace/ -e WANDB_API_KEY=321d12190ddd47a8f33f5c2b00bfe36f25691501 --runtime=nvidia --name deep deeplearning:latest
```
