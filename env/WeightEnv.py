import random
import gym
import numpy as np

M = 5.0
T = 1.0
GOAL = 0.001

class WeightEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(WeightEnv, self).__init__()
        self.reward_range = (-float('inf'), 0.0)
        self.state = np.array([0, 0, 0]) # position, velocity, acceleration

        # action: force[-10, 10]
        self.action_space = gym.spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)

        # observation: position[-10,10], velocity[-10,10], acceleration[-10,10], jerk[-10,10]
        self.observation_space = gym.spaces.Box(np.array([-10, -10, -10, -10]), np.array([10, 10, 10, 10], dtype=np.float32))

        self.steps = 0

    def step(self, action):
        prev_position = self.state[0]
        prev_velocity = self.state[1]
        prev_acceleration = self.state[2]
        action_force = min(max(action[0], -10.0), 10.0)

        next_acceleration = action_force / M
        next_jerk = next_acceleration - prev_acceleration
        next_velocity = prev_velocity + next_acceleration * T
        next_position = prev_position + next_velocity * T

        self.steps += 1
        done = ((abs(next_position) < GOAL) and (abs(next_velocity) < GOAL)) or (self.steps > 100)
        self.state = np.array([next_position, next_velocity, next_acceleration])
        reward = 0.0 - (abs(next_position)**2) - (abs(next_velocity)**2) - (abs(next_acceleration)**2) - (abs(next_jerk)**2)
        return np.array([next_position, next_velocity, next_acceleration, next_jerk]), reward, done, {}
        
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.steps = 0
        self.state = np.array([self.np_random.uniform(low=-10.0, high=10.0), 0, 0]) # position, velocity, accel
        return np.array([self.state[0], self.state[1], self.state[2], 0])
