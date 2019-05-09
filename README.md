# Habitat-TF
PPO agent for RL environments

## Contents

- [Environments]()
- [Training]()
- [Codebase structure](#structure)
- [Acknowledgements](#acknowledgements)
## Structure

`python train.py --hparams ppo_sample --sys local`

```
Habitat-TF/
├── README.md
├── requirements.txt
├── rl
│   ├── agents
│   │   ├── agent.py
│   │   ├── algos
│   │   │   ├── action_function
│   │   │   │   ├── basic.py
│   │   │   │   ├── __init__.py
│   │   │   │   └── registry.py
│   │   │   ├── compute_gradient
│   │   │   │   ├── basic.py
│   │   │   │   ├── __init__.py
│   │   │   │   ├── registry.py
│   │   │   │   └── utils.py
│   │   │   ├── gibson_ppo.py
│   │   │   ├── __init__.py
│   │   │   ├── ppo.py
│   │   │   ├── registry.py
│   │   │   └── utils.py
│   │   ├── __init__.py
│   │   └── registry.py
│   ├── envs
│   │   ├── configs
│   │   │   ├── baselines
│   │   │   ├── datasets
│   │   │   │   └── pointnav
│   │   │   ├── tasks
│   │   │   └── test
│   │   ├── env.py
│   │   ├── gibson.py
│   │   ├── gym_env.py
│   │   ├── habitat_env
│   │   │   └── gibson.py
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── reward_augmentation.py
│   │   └── utils.py
│   ├── hparams
│   │   ├── defaults.py
│   │   ├── gibson_ppo.py
│   │   ├── __init__.py
│   │   ├── ppo.py
│   │   ├── registry.py
│   │   └── utils.py
│   ├── __init__.py
│   ├── memory
│   │   ├── __init__.py
│   │   ├── memory.py
│   │   ├── registry.py
│   │   └── simple.py
│   ├── models
│   │   ├── basic
│   │   │   ├── basic.py
│   │   │   └── __init__.py
│   │   ├── gibson_model
│   │   │   ├── gibson_model.py
│   │   │   └── __init__.py
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── registry.py
│   └── utils
│       ├── checkpoint.py
│       ├── flags.py
│       ├── __init__.py
│       ├── logger.py
│       ├── lr_schemes.py
│       ├── rand.py
│       ├── sys.py
│       └── utils.py
└── train.py

```
## Acknowledgments

- [for.ai](https://for.ai) 
- Adapted from [this repo](https://github.com/for-ai/rl)