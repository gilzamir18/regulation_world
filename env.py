from typing import Optional
import random
import gymnasium as gym
import numpy as np

class HomeostaticEnv(gym.Env):
    def __init__(self, num_vars: int=2, rewarding: str = "default", reward_scale: float = 1.0, max_steps:int=1000, random_safety_zone=False, decay_std: float = 0.0, **kwargs):
        self.num_vars = num_vars
        self.decay_std = decay_std
        self.state = [0.0] * num_vars  # Initialize all variables to 0.0
        self.survival_zone = [0.0] * num_vars
        self.target_values = [0.0] * num_vars
        self.decay_ratio = [0.01] * num_vars  # Default decay ratio for each variable
        self.action_scaler = 0.2
        self.safety_zone = [1.0] * num_vars
        self.prev_state = self.state.copy()
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.step_count = 0
        self.random_safety_zone = random_safety_zone

        self.observation_space = gym.spaces.Box(
            low=-1.5,
            high=1.5,
            shape=(self.num_vars * 3,),
            dtype=float
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_vars,),
            dtype=float
        )

        self.rewarding = rewarding

        
    def _get_obs(self):
        return np.asarray(self.state + self.survival_zone + self.safety_zone, dtype=float)
    

    def _get_info(self):
        return {
            "target_values": self.target_values,
            "decay_ratio": self.decay_ratio
        }


    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)
        random.seed(seed)
        self.state = [ random.uniform(-0.9, 0.9) for _ in range(self.num_vars) ]
        self.survival_zone = [1.0] * self.num_vars
        self.target_values = [0.0] * self.num_vars
        self.decay_ratio = [0.01] * self.num_vars
        if self.random_safety_zone:
            self.safety_zone = [random.uniform(0.5, 0.9) for _ in range(self.num_vars)]
        else:
            self.safety_zone = [0.5 for _ in range(self.num_vars)]
        self.prev_state = self.state.copy()
        self.step_count = 0
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        if self.rewarding == "QD":
            return self._qd_step(action)
        elif self.rewarding == "operational_regimes" or self.rewarding == "OR_full":
            return self._operationalregimes_step(action)
        elif self.rewarding == "OR_no_safety_zone":
            return self._or_no_safety_zone_step(action)
        elif self.rewarding == "OR_unbounded":
            return self._or_unbounded_step(action)
        elif self.rewarding == "OR_no_maintenance_reward":
            return self._or_no_maintenance_reward_step(action)
        else:
            return self._default_step(action)

    def _env_dynamic(self, action):
        self.prev_state = self.state.copy()
        if self.decay_std > 0.0:
            self.decay_ratio = [dr + random.gauss(0.0, self.decay_std) for dr in self.decay_ratio]
        self.state = [ (self.state[i] - self.decay_ratio[i]) for i in range(self.num_vars) ] 
        self.state = [ self.state[i] + a * self.action_scaler for i, a in enumerate(action) ]
        self.step_count += 1
        return any(abs(self.state[i] - self.target_values[i]) > self.survival_zone[i] for i in range(self.num_vars))
       
    def _default_step(self, action):
        action = np.clip(action, 0, 1)
        done = self._env_dynamic(action)
        ended = self.step_count > self.max_steps      
        distance = sum((self.state[i] - self.target_values[i])**2 for i in range(self.num_vars))
        reward = -distance * self.reward_scale
        return self._get_obs(), reward, done, ended, self._get_info()
    
    def _qd_step(self, action):
        action = np.clip(action, 0, 1)
        done = self._env_dynamic(action)
        ended = self.step_count > self.max_steps
        dt = sum((self.state[i] - self.target_values[i]) ** 2 for i in range(self.num_vars))
        dt_1 = sum((self.prev_state[i] - self.target_values[i]) ** 2 for i in range(self.num_vars))
        reward = (dt_1 - dt) * self.reward_scale
        return self._get_obs(), reward, done, ended, self._get_info()
    
    def _operationalregimes_step(self, action):
        action = np.clip(action, 0, 1)
        done = self._env_dynamic(action)
        ended = self.step_count > self.max_steps
        reward = 0.0
        for i in range(self.num_vars):
            dt = abs(self.state[i] - self.target_values[i])
            dt_1 = abs(self.prev_state[i] - self.target_values[i])
            u = (self.state[i] <= self.safety_zone[i])
            v = (dt_1 <= dt)
            reward += (u + (1 - u) * (1 - 3 * v))
        return self._get_obs(), reward * self.reward_scale, done, ended, self._get_info()

    def _or_no_safety_zone_step(self, action):
        action = np.clip(action, 0, 1)
        done = self._env_dynamic(action)
        ended = self.step_count > self.max_steps
        reward = 0.0
        for i in range(self.num_vars):
            dt = abs(self.state[i] - self.target_values[i])
            dt_1 = abs(self.prev_state[i] - self.target_values[i])
            v = (dt_1 <= dt) # True if worsened or stayed same
            # No safety zone: 1 if improved, -2 if worsened
            reward += (1 - 3 * v)
        return self._get_obs(), reward * self.reward_scale, done, ended, self._get_info()

    def _or_unbounded_step(self, action):
        action = np.clip(action, 0, 1)
        done = self._env_dynamic(action)
        ended = self.step_count > self.max_steps
        reward = 0.0
        for i in range(self.num_vars):
            dt = abs(self.state[i] - self.target_values[i])
            dt_1 = abs(self.prev_state[i] - self.target_values[i])
            u = (self.state[i] <= self.safety_zone[i])
            v = (dt_1 <= dt)
            
            if u:
                reward += 1.0
            else:
                # Magnitude scales with distance from safety zone
                dist_from_safety = dt - self.safety_zone[i]
                if not v: # improved
                    reward += 1.0 * (1.0 + dist_from_safety)
                else: # worsened
                    reward += -2.0 * (1.0 + dist_from_safety)
                    
        return self._get_obs(), reward * self.reward_scale, done, ended, self._get_info()

    def _or_no_maintenance_reward_step(self, action):
        action = np.clip(action, 0, 1)
        done = self._env_dynamic(action)
        ended = self.step_count > self.max_steps
        reward = 0.0
        for i in range(self.num_vars):
            dt = abs(self.state[i] - self.target_values[i])
            dt_1 = abs(self.prev_state[i] - self.target_values[i])
            u = (self.state[i] <= self.safety_zone[i])
            v = (dt_1 <= dt)
            
            if not u:
                # Outside safety zone: 1 if improved, -2 if worsened
                reward += (1 - 3 * v)
            # If inside safety zone, reward is 0 (u -> +0 instead of +1)
            
        return self._get_obs(), reward * self.reward_scale, done, ended, self._get_info()
    
    def render(self):
        print(f"State: {self.state}, Target: {self.target_values}, Survival Zone: {self.survival_zone}")