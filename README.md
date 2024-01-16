# safe-reinforcement-learning
Repository containing the Safe Reinforcement Learning team's work.

## Docker instructions

The `saferl/tensorflow` and `saferl/pytorch` Docker images are based on Nvidia's `nvcr.io/nvidia/tensorflow:23.10-tf2-py3` and `nvcr.io/nvidia/pytorch:23.11-py3` Docker images, respectively.
We have added additional pre-installed dependencies to these images, such as `gymnasium`.
To run containers based on these images, run one of the following commands with your working directory set to the [root of this repository](/).
```bash
# saferl/tensorflow
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd):/workspace/safe-reinforcement-learning --rm saferl/tensorflow

# saferl/pytorch
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd):/workspace/safe-reinforcement-learning --rm saferl/pytorch
```