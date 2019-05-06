# DRLND-Project-Collaboration-and-Competition

Project 3: Collaboration and Competition
Introduction
For this project, you will work with the Tennis environment.

Trained Agent

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

Getting Started
Download the environment from one of the links below. You need only select the environment that matches your operating system:

Linux: click here
Mac OSX: click here
Windows (32-bit): click here
Windows (64-bit): click here
(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

Place the file in the DRLND GitHub repository, in the p3_collab-compet/ folder, and unzip (or decompress) the file.

Instructions
Follow the instructions in Tennis.ipynb to get started with training your own agent!

(Optional) Challenge: Crawler Environment
After you have successfully completed the project, you might like to solve the more difficult Soccer environment.

Soccer

In this environment, the goal is to train a team of agents to play soccer.

You can read more about this environment in the ML-Agents GitHub here. To solve this harder task, you'll need to download a new Unity environment. (Note: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:

Linux: click here
Mac OSX: click here
Windows (32-bit): click here
Windows (64-bit): click here
Then, place the file in the p3_collab-compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file. Next, open Soccer.ipynb and follow the instructions to learn how to use the Python API to control the agent.

(For AWS) If you'd like to train the agents on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agents without enabling a virtual screen, but you will be able to train the agents. (To watch the agents, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)
