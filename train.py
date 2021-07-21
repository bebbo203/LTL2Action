
import gym
import torch as th

from envs.adversarial import AdversarialEnv9x9
from custom_policy import CustomActorCriticPolicy

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from customcallback import CustomCallback


env = AdversarialEnv9x9()
model = PPO(CustomActorCriticPolicy, env, verbose=1, tensorboard_log="./tensorboard", device="cuda")

# Load pre-trained weights for the LTL module
# model.policy.mlp_extractor.rnn.load_state_dict(th.load("./weights_rnn.pt"))
# model.policy.mlp_extractor.ltl_embedder.load_state_dict(th.load("./weights_ltl.pt"))

# Callback function to save the best model
eval_callback = CustomCallback(Monitor(env), min_ep_rew_mean=.5, n_eval_episodes=20,
                               best_model_save_path='./logs/', log_path='./logs/',
                               eval_freq=4*2048, verbose=1, render=False)

# Training
# model.learn(int(5e6), callback=eval_callback)
model.learn(2048*5, callback=eval_callback)

# Evaluation
mean_rew, std_rew = evaluate_policy(model.policy, Monitor(env),
                                    n_eval_episodes=20,
                                    render=False,
                                    deterministic=True)
print(f"Mean reward: {mean_rew:.2f} +/- {std_rew:.2f}")

