To build an image from one of the Dockerfiles in this directory, run one of the following commands with your working directory set to the [root of this repository](/).
```bash
# saferl/tensorflow
docker build -t saferl/tensorflow -f ./docker/tensorflow.Dockerfile ./docker

# saferl/pytorch
docker build -t saferl/pytorch -f ./docker/pytorch.Dockerfile ./docker
```

Make sure to occasionally run `docker image prune` to remove dangling images that may have been created in the process of rebuilding `saferl/tensorflow` and `saferl/pytorch`.
Dangling images are denoted by `<none>` in the list displayed after running `docker images`.