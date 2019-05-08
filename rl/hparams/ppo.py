from .defaults import default
from .registry import register


@register
def ppo():
  hps = default()
  hps.memory = "SimpleMemory"
  hps.models = ["PPOActor", "PPOCritic"]
  hps.agent = "PPO"
  hps.lr = {'actor_lr': 0.0001, 'critic_lr': 0.0002}
  hps.lr_decay = {'actor_lr': 'no_decay', 'critic_lr': 'no_decay'}
  hps.batch_size = 32
  hps.num_steps = 128
  hps.num_epochs = 10
  hps.hidden_size = 100
  hps.gamma = 0.9
  hps.memory_size = 50000
  hps.action_function = "uniform_random_action"
  hps.grad_function = "ppo"
  hps.normalize_reward = False
  hps.clipping_coef = 0.2

  return hps


@register
def ppo_cartpole():
  hps = ppo()
  hps.env = "CartPole-v1"
  hps.gamma = 0.99
  return hps


@register
def ppo_mountaincar():
  hps = ppo()
  hps.env = "MountainCar-v0"
  hps.reward_augmentation = "mountain_car_default"
  return hps


@register
def ppo_pong():
  hps = ppo()
  hps.env = "Pong-v0"
  return hps
