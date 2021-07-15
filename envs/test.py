from adversarial import AdversarialEnv9x9
from matplotlib import pyplot as plt


e = AdversarialEnv9x9()
e.reset()

for i in range(1000):
    action = e.action_space.sample()
    o, r, d, _ = e.step(action)
    # print(e.get_events())
    e.render()
    if d:
        e.reset()
