from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
from env import HomeostaticEnv

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=HomeostaticEnv,
    max_episode_steps=1000,  # Prevent infinite episodes
)

num_tests = 1
num_vars = 100
if __name__=="__main__":
    sufix = f"nv{num_vars}"
    env_args = {"num_vars":num_vars, "rewarding":"euclidian", "reward_scale":1.0}
    env = make_vec_env("gymnasium_env/GridWorld-v0", n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs=env_args)
    for i in range(num_tests):
        model = PPO("MlpPolicy", env, device="cpu", verbose=1, tensorboard_log="./ppo_homeostatic_tensorboard/")
        model.learn(total_timesteps=250_000, tb_log_name=f"ppo_euclidian_{sufix}")