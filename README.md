# Habitat-TF
PPO agent for RL environments. This code is an adaptation of [FOR.ai's RL repository](#acknowledgments)


- [Habitat-TF](#habitat-tf)
  - [Environments](#environments)
  - [Training](#training)
  - [Installation](#installation)
  - [Structure](#structure)
  - [Acknowledgments](#acknowledgments)
  - [References](#references)


## Environments

- Gym environments (Cartpole, MountainCar, Pong etc)
- Gibson 3D Photo-realistic environment

## Training

Environment
  - Cartpole:  `python train.py --hparams ppo_cartpole --sys local` 
  - Pong: `python train.py --hparams ppo_pong --sys local`
  - MountainCar: `python train.py --hparams ppo_mountaincar --sys local`
  - Gibson: `python train.py --hparams ppo_gibson --sys local`

To train asynchronously, set `--num_workers` flag to the number of worker threads.

A complete list of hyperparameters used for each environment can be found [here for gym envs](rl/hparams/ppo.py) and [here for Gibson](rl/hparams/gibson_ppo.py)
## Installation

`pip install -r requirements.txt`

To be able to train on the Gibson environment, you would need to install the following packages

- Install `habitat_sim` from [here](https://github.com/facebookresearch/habitat-sim)
- Install `habitat_api` from [here](https://github.com/facebookresearch/habitat-api)
- Download the data from [here](https://github.com/facebookresearch/habitat-api#task-datasets) and follow the structure mentioned on the repository
- Place the data in a folder named `data` under `Habitat-TF/`

## Structure

```
rl/
├── agents
│   ├── agent.py
│   ├── algos
│   │   ├── action_function
│   │   │   ├── basic.py
│   │   │   ├── __init__.py
│   │   │   └── registry.py
│   │   ├── compute_gradient
│   │   │   ├── basic.py
│   │   │   ├── __init__.py
│   │   │   ├── registry.py
│   │   │   └── utils.py
│   │   ├── gibson_ppo.py
│   │   ├── __init__.py
│   │   ├── ppo.py
│   │   ├── registry.py
│   │   └── utils.py
│   ├── __init__.py
│   └── registry.py
├── envs
│   ├── configs
│   │   ├── baselines
│   │   ├── datasets
│   │   │   └── pointnav
│   │   ├── tasks
│   │   └── test
│   ├── env.py
│   ├── gibson.py
│   ├── gym_env.py
│   ├── habitat_env
│   │   └── gibson.py
│   ├── __init__.py
│   ├── registry.py
│   ├── reward_augmentation.py
│   └── utils.py
├── hparams
│   ├── defaults.py
│   ├── gibson_ppo.py
│   ├── __init__.py
│   ├── ppo.py
│   ├── registry.py
│   └── utils.py
├── __init__.py
├── memory
│   ├── __init__.py
│   ├── memory.py
│   ├── registry.py
│   └── simple.py
├── models
│   ├── basic
│   │   ├── basic.py
│   │   └── __init__.py
│   ├── gibson_model
│   │   ├── gibson_model.py
│   │   └── __init__.py
│   ├── __init__.py
│   ├── model.py
│   └── registry.py
└── utils
    ├── checkpoint.py
    ├── flags.py
    ├── __init__.py
    ├── logger.py
    ├── lr_schemes.py
    ├── rand.py
    ├── sys.py
    └── utils.py


```
## Acknowledgments

- [for.ai](https://for.ai) 
- [Piotr Kozakowski](https://github.com/koz4k)
- Adapted from [this repo](https://github.com/for-ai/rl)

## References

- Habitat [paper](https://arxiv.org/abs/1904.01201)
- FAIR habitat repositories
  - [Habitat API](https://github.com/facebookresearch/habitat-api)
  - [Habitat Simulator](https://github.com/facebookresearch/habitat-sim)
  - [Habitat Challenge](https://github.com/facebookresearch/habitat-challenge)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
