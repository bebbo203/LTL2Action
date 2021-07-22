
import os
import time
import torch

from envs.adversarial import AdversarialEnv9x9
from envs.myopic import AdversarialMyopicEnv9x9

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


MANUAL_CONTROL = False
MYOPIC = False
MODEL = "main" # {'main' | 'toy_grid' | 'myopic'}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def control(letter):
    ''' Helper function to manually control the agent. '''
    if letter == 'a':   return 0 # sx
    elif letter == 'd': return 1 # dx
    elif letter == 'w': return 2 # forward
    else:               return 3 # no move


env = AdversarialEnv9x9() if not MYOPIC else AdversarialMyopicEnv9x9()
model = PPO.load(os.path.join("logs", MODEL), device=DEVICE) if not MANUAL_CONTROL else None

obs = env.reset()
for i in range(10000):
    action = control(input()) if MANUAL_CONTROL else model.predict(obs)[0]
    obs, rew, done, _ = env.step(action)
    # print(f"obs: {env.get_events()}")
    env.render()
    time.sleep(.25)
    if done:
        env.reset()

