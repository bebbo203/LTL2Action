from adversarial import AdversarialEnv
from minigrid_env import MinigridEnv
from matplotlib import pyplot as plt

e = AdversarialEnv(size=9)
e = MinigridEnv(e, "abc")

e.reset()

for i in range(1000000):
    o, _, _, _ = e.step(e.action_space.sample())
    print(e.get_events())
    e.env.render()

