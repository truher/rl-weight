import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from env.WeightEnv import WeightEnv

def printit(x,y):
    print(f'{x[0]:10.3f} {x[1]:10.3f} {x[2]:10.3f} {x[3]:10.3f} {y:10.3f}')

def main():
    env = DummyVecEnv([lambda: WeightEnv()])
    env.env_method("seed", 0)
    model = PPO2(MlpPolicy, env, tensorboard_log="/tmp/foo")
    model.learn(total_timesteps=1000000)
    obs = env.reset()
    print('position   velocity   accel      jerk       reward')
    for i in range(2000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        if done:
            printit(info[0]['terminal_observation'], rewards[0])
            print("")
        printit(obs[0], rewards[0])

if __name__ == '__main__':
    main()
