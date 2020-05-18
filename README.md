[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"

# Deep Reinforcement Learning Project: Collaboration and Competition

## Project Details
This project aims to train two agents to control rackets to bounce a ball over a net in a virtual environment (Tennis environment) for as many time steps as possible. Thus, the goal is to keep the ball in play as long as possible.

The trained agent in this project is a Deep Deterministic Policy Gradient (DDPG) based agent.

![Trained Agent][image1]
### The Environment
The environment chosen for the project is similar but not identical to the version of the  [Tennis Environment](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis)  from the Unity ML-Agents toolkit.
* This environment contains 2 identical agents, each with its own copy of the environment.  

* In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

- The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. 
-  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

- The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

	- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the **maximum** of these 2 scores.
	- This yields a single **score** for each episode.

- The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5. 

## Getting Started
### Downloading the environment
1. Download the environment from one of the links below. You need only select the environment that matches your operating system:

Platform | Link
-------- | -----
Linux             | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
Mac OSX           | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
Windows (32-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
Windows (64-bit)  | [Link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
* (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

* (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Unzip (or decompress) the downloaded file and store the path of the executable as we will need the path to input on `Tennis.ipynb`. 
### Dependencies
Please follow the instructions on [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install the required libraries.
## Instructions
Follow the instructions in `Tennis.ipynb` to get started with training your own agent!