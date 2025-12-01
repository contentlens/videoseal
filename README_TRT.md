# Exporting models to TRT
Currently only the embedder is supported for exporting. The following steps will take you through on how to create the trt model. 

Very important: You need a system with the same configuration as the system the TRT model was generated in to make the model run.

## Pre-requisites
- Docker
- nvidia-container-toolkit
- CUDA 12+
- NVIDIA Driver 535+

## Steps
- Download the repository
- Create the docker image
- Run the container and create the models.

### Download the repository
```
git clone https://github.com/contentlens/videoseal
```

### Create the docker image
```
sudo docker build -t converter -f . convert.Dockerfile
```

### Run the container and create the models
Create a folder called `models` before running the container to use it as a volume to write the model into.
```
sudo docker run -itv ./models:/app/models --gpus all converter:latest bash
```
Once you bash into it, run the following command to create the model:
```
source ./venv/bin/activate
python3 export_detector_trt.py
```

The model will be created in `models` folder.
