# LTL2Action: Multi-task Reinforcement Learning using LTL 
The repository implements a reinforcement learning algorithm to address the problem of instruct RL agents to learn temporally extended goals in multi-task environments. The project is based on the idea introduced in [LTL2Action: LTL Instruction for multi-task RL](https://arxiv.org/pdf/2102.06858.pdf) (Vaezipoor et al. 2021).

More details can be found in the **[Report](https://github.com/bebbo203/LTL2Action/blob/main/report.pdf)**.
The full implementation of the project is available on **[Github](https://github.com/bebbo203/LTL2Action)**.

## Environment

The environment is implemented with [gym-minigrid](https://github.com/maximecb/gym-minigrid): an agent (red triangle) must navigate in a 7x7 map. There are walls (grey squares), goals (colored squares) that are shuffled at each episode and doors (blue rectangles). The actions are _go straight_, _turn left_, _turn right_ and the observations returned are the 7x7 colored grid and the orientation of the agent codified by an integer.

<div style="text-align:center"><img src="https://github.com/bebbo203/LTL2Action/blob/main/imgs/env.png?raw=true" width="300" height="300"></div>


## Framework

We implemented a RL framework with LTL instructions which learn to solve complex tasks (formalized in LTL language) in challenging environments. At every iteration the RL agent can partially observe the environment sorrounding it and through an event detector a set of truth assignments which are going to progressed (through a progression function) the LTL instruction, identifying the remaining aspect of the tasks to be accomplished.
Therefore, the overall method relies on two modules which serve as feature extractors: one for the observation of the environment and one for the LTL instruction, which are later combined together to forms the input of a standard RL algorithm (PPO).

<div style="text-align:center"><img src="https://github.com/bebbo203/LTL2Action/blob/main/imgs/modules.png?raw=true" width="500" height="300"></div>


## Results

The method is able to solve multi-task environments with an high success rate.
To make the agent generalize over the task formulas, every episode is descripted with a different LTL task sampled from a finite set. 
In the following plot, a normal PPO agent and the novel agent are trained on the same environment.
The two task taken in consideration are: 

* go to *blue* THEN go to *red*
* go to *blue* THEN go to *green*

A Myopic agent reach a success rate of 50%, meaning that it cannot "see" what is the successive goal after the blue one.

<div style="text-align:center"><img src="https://github.com/bebbo203/LTL2Action/blob/main/imgs/ep_rew_mean.png?raw=true" width="450" height="350"></div>


## Experiments

The agent is trained over a variety of LTL tasks, like partially ordered tasks and avoidance tasks.
In the gif below the task *"eventually go to blue square and then go to green square"* is performed.

<div style="text-align:center"><img src="https://github.com/bebbo203/LTL2Action/blob/main/imgs/openaigym.video.0.gif?raw=true" width="350" height="350"></div>

In the second example video the agent must execute a sequence of partially ordered task appearing in the image bottom part, showing also the **progression** mechanism. When the task is accomplished the LTL formula progresses to *true*. Note that LTL formulae are represented in *prefix notation* by using tokens for operators and prepositions and brackets for relations.

<div style="text-align:center"><img src="https://github.com/bebbo203/LTL2Action/blob/main/imgs/video.gif?raw=true" width="450" height="450"></div>
<br>

## Presentation

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vT0ZKWbtYdSO2DeomSDiGrbZiNMN4GdYaB4Cww3pqnQNOfCpL531-f9rtMOHbvjfmHZT-zb0C2CV1jU/embed?start=false&loop=true&delayms=3000" frameborder="0" width="960" height="569" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>
<br>

## Installation

### Requirements
* [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/)
* [gym-minigrid](https://github.com/maximecb/gym-minigrid)

or, for better compatibility, the following command can be used:

### Set up the conda environment
```
conda env create -f environment.yml
```

## How to Use

### Training RL agent

```
python train.py
```

### Test

```
python test.py
```

## References
- Vaezipoor, Pashootan, Li, Andrew, Icarte, Rodrigo Toro, and McIlraith, Sheila (2021). “LTL2Action:Generalizing LTL Instructions for Multi-Task RL”. In:International Conference on MachineLearning (ICML)
- Icarte, Rodrigo Toro, Klassen, Toryn Q., Valenzano, Richard Anthony, and McIlraith, Sheila A.(2018). “Teaching Multiple Tasks to an RL Agent using LTL.” In:International Conferenceon Autonomous Agents and Multiagent. Systems (AAMAS), (pp. 452–461)
