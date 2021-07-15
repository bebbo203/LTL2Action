
from adversarial import AdversarialEnv9x9


def vocabulary(letter):
    # manual control
    if letter == 'a':   return 0 # sx
    elif letter == 'd': return 1 # dx
    elif letter == 'w': return 2 # forward
    else: raise NotImplementedError

e = AdversarialEnv9x9()
e.reset()

for i in range(1000):
    # action = e.action_space.sample()
    action = vocabulary(input())
    o, r, d, _ = e.step(action)
    # print(e.get_events())
    e.render()
    if d:
        e.reset()
