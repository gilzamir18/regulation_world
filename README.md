# Homeostatic Environment

**Class**: `HomeostaticEnv`

The `HomeostaticEnv` is a continuous control environment that simulates a homeostatic regulation problem with multiple resources. The agent must maintain a set of simulated physiological variables (resources) within specific ranges (Safety and Survival zones) while combating a constant natural decay rate.

The environment challenges the agent to learn a policy that balances multiple variables simultaneously. Failure to maintain any single variable within the survival zone results in the immediate termination of the episode (simulated "death").

## Action Space

The action space is a continuous `Box`.

* **Type**: `Box(low=-1.0, high=1.0, shape=(num_vars,), dtype=float)`
* **Description**: Each element in the vector represents the intensity of the corrective action applied to the corresponding variable.
* **Dynamics**: The environment applies a fixed **decay** to all variables at every step, and then adds the scaled action.


*  (decay ratio): Fixed at `0.01`.
*  (action scaler): Fixed at `0.2`.



## Observation Space

The observation is a flat continuous vector containing the current state of variables and their respective limits.

* **Type**: `Box(low=-1.5, high=1.5, shape=(3 * num_vars,), dtype=float)`
* **Structure**: The vector is a concatenation of three segments:
1. **Current State** : The current value of each variable.
2. **Survival Zone** : The absolute limit (threshold) for survival (default: `1.0`).
3. **Safety Zone** : The ideal threshold where the agent should aim to keep the variables (default: `0.5` or randomized).

## Rewards

The environment supports multiple reward functions, selected via the `rewarding` argument during initialization. The final reward is always multiplied by `reward_scale` (default `1.0`).

Let  be the sum of squared distances to the target (0.0) at step : .

### 1. `default`

Penalizes the agent based on the total quadratic distance from the equilibrium point (0.0).


### 2. `euclidian`

Rewards the agent for reducing the total distance to the target compared to the previous step (potential-based reward).

### 3. `operational_regimes`

Calculates rewards individually for each variable and sums them.

* Distances calculated relative to the **Target (0.0)**.

For each variable $i$:

*  $u_i = 1$ if $|s_i| \le \text{safety\_zone}_i$, else $0$.
*  $v_i = 1$ if distance worsened ($d_{prev} \le d_{curr}$), else $0$ (improved)..

$$R = \sum_{i=0}^{N} [u_i + (1-u_i)(1 - 3v_i)] \times \text{scale}$$

*Logic*: If inside the safety zone, +1. If outside but improving, +1. If outside and worsening, -2.

## Starting State

When `reset()` is called:

1. **State**: Each variable is initialized uniformly between .
2. **Survival Zone**: Fixed at `1.0`.
3. **Target Values**: Fixed at `0.0`.
4. **Safety Zone**:
* If `random_safety_zone=False`: Fixed at `0.5`.
* If `random_safety_zone=True`: Randomized uniformly between  per variable.

## Episode End

The episode terminates (`done=True`) if:

1. **Homeostatic Failure**: The absolute value of **any** variable exceeds its survival zone.

The episode is truncated (`ended=True`) if:

1. **Time Limit**: The step count exceeds `max_steps` (default `1000`).

## Arguments

Arguments to be passed to `gym.make` or the class constructor:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `num_vars` | `int` | `2` | The number of homeostatic variables to control. |
| `rewarding` | `str` | `"default"` | The reward function to use (`"default"`, `"euclidian"`, `"operational_regimes"`, `"default"`). |
| `reward_scale` | `float` | `1.0` | Scalar to multiply the calculated reward. |
| `max_steps` | `int` | `1000` | Maximum steps per episode before truncation. |
| `random_safety_zone` | `bool` | `False` | If `True`, the safety zone thresholds are randomized on reset. |

## Usage Example

```python
import gymnasium as gym
from env import HomeostaticEnv

# Register the environment (if not using direct class instantiation)
gym.register(
    id="HomeostaticEnv-v0",
    entry_point=HomeostaticEnv,
    max_episode_steps=1000,
)

# Initialize
env = gym.make(
    "HomeostaticEnv-v0", 
    num_vars=20, 
    rewarding="euclidian", 
    reward_scale=0.1,
    random_safety_zone=True
)

obs, info = env.reset()
done = False
truncated = False

while not (done or truncated):
    # Sample a random action
    action = env.action_space.sample()
    
    # Step the environment
    obs, reward, done, truncated, info = env.step(action)
    
    if done:
        print("Episode finished (Homeostatic Failure)")
    elif truncated:
        print("Episode finished (Max Steps Reached)")

env.close()

```

# 📚 Citation
If you use Homeostatic Environment in your work please cite it as follows:
```
@misc{gilzamir2025homeoenv,
    author={Gilzamir Gomes},
    title = {HomeostaticEnv: A Multi-Resource Environment for Homeostatic Reinforcement Learning Experiments},
    url = {https://gilzamir18.github.io/preprints/}
    year = {2025}
}
```