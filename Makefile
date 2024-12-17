SHELL = /bin/bash

# Docker image
DOCKER_TAG=mri-guided-diffusion:latest

# Args
WEIGHTS_PATH=
DATA_PATH=
SAVE_DIR=
PREPROCESS_ARGS=
INPAINT_ARGS=
TRAIN_ARGS=
GPU_ID=

.PHONY: get_weights build preprocess_mri train inpaint clean sample

# Default rule
all: get_weights build preprocess_mri inpaint

get_weights:
	python -u -m get_weights $(SAVE_DIR)

build:
	docker build --no-cache . -f Dockerfile -t $(DOCKER_TAG)

preprocess_mri:
	docker run \
    -v $(DATA_PATH):/app/data/new/raw \
    -v $(SAVE_DIR):/app/data/new/processed \
    $(DOCKER_TAG) python -m masked_diffusion.etl.preprocess_mri $(PREPROCESS_ARGS)


inpaint:
	docker run \
	-v $(DATA_PATH):/app/data/new/processed \
    -v $(SAVE_DIR):/app/data/new/processed \
    -v $(WEIGHTS_PATH):/app/masked_diffusion/model/pretrained \
    -m 8g \
	-e WANDB_API_KEY= \
    $(DOCKER_TAG) python -u -m masked_diffusion.model.inpaint $(INPAINT_ARGS)


train:
	docker run \
		-m 8g \
		-e WANDB_API_KEY=57df528e0432d36b1f85bb21edd76a6ae9a1bcba \
		$(DOCKER_TAG) python -u -m masked_diffusion.model.train $(TRAIN_ARGS)

shell:
	docker run -it \
	    $(DOCKER_TAG) /bin/bash
