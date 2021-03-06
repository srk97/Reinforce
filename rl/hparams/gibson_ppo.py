from .defaults import default
from .registry import register


@register
def ppo():
  hps = default()
  hps.state_processor = "GibsonPixelProcessor"
  hps.models = ["GibsonPPOActor", "GibsonPPOCritic"]
  hps.agent = "Gibson_PPO"
  hps.clip_grad_norm = True
  hps.num_steps = 10
  hps.batch_size = 10
  hps.max_grad_norm = 0.5
  hps.value_loss_coef = 0.5
  hps.lr = {'actor_lr': 2.5e-4, 'critic_lr': 2.5e-4}
  hps.lr_decay = {'actor_lr': 'no_decay', 'critic_lr': 'no_decay'}
  hps.hidden_size = 512
  hps.num_epochs = 10
  hps.gamma = 0.99
  hps.sensors = ["RGB_SENSOR", "DEPTH_SENSOR"]
  hps.task_config = 'tasks/pointnav.yaml'
  hps.memory_size = 50000
  hps.action_function = "uniform_random_action"
  hps.grad_function = "ppo"
  hps.normalize_reward = True
  hps.input_goal_size = 2
  hps.clipping_coef = 0.1
  hps.kernel_sizes = [8, 4, 3]
  hps.strides = [4, 2, 1]
  hps.state_shape = [256, 256, 4]
  hps.num_actions = 4
  hps.pixel_input = True

  return hps


@register
def ppo_gibson():
  hps = ppo()
  hps.env = "gibson_env"
  return hps
