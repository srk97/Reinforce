from .defaults import default
from .registry import register


@register
def ppo():
  hps = default()
  hps.state_processor = "GibsonPixelProcessor"
  hps.models = {"actor": "GibsonPPOActor", "critic": "GibsonPPOCritic"}
  hps.agent = "PPO"
  hps.clip_grad_norm = True
  hps.n_steps = 128
  hps.max_grad_norm = 0.5
  hps.value_loss_coef = 0.5
  hps.actor_lr = 1e-5
  hps.critic_lr = 1e-5
  hps.batch_size = 32
  hps.hidden_size = 512
  hps.gamma = 0.99
  hps.sensors = ["RGB_SENSOR", "DEPTH_SENSOR"]
  hps.task_config = 'tasks/pointnav.yaml'
  hps.memory_size = 50000
  hps.action_function = "uniform_random_action"
  hps.grad_function = "ppo"
  hps.num_actor_steps = 5
  hps.num_critic_steps = 5
  hps.normalize_reward = True
  hps.input_goal_size = 2
  hps.clipping_coef = 0.1
  hps.kernel_sizes = [8, 4, 3]
  hps.strides = [4, 2, 1]
  hps.state_shape = [256, 256, 4]
  hps.num_actions = 4
  hps.pixel_input = True
  hps.mode = "train"

  return hps


@register
def ppo_sample():
  hps = ppo()
  hps.env = "gibson_env"
  return hps
