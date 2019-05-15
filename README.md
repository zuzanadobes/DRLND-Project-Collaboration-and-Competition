# DRLND-Project-Collaboration-and-Competition

Draft submission to Udacity of project code towards the completion of the nano degree "Deep Reinforcement Learning" (DRL). Project #3 called "Collaboration and Competition" trains an agent to play tennis,  and to keep the ball "in play" as long as possible,

# Introduction

This project works with the Tennis environment provided by Udacity.  This project was completed on a local Windows machine but it can also be run in the Udacity Workspace. Instructions on how to download and setup Unity ML environments can be found in [ Unity ML-Agents Github repo ] https://github.com/Unity-Technologies/ml-agents. The project environment is similar to the Tennis environment on the [ Unity ML-Agents GitHub page ] https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md.

# Trained Agent

In the loaded in environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If (one of the two) agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, the rewards that each agent received (without discounting) are summed up, which becomes's the agent's score. This results in 2 scores, the max of which is recorded as the episode score.  The environment is considered solved, when the average (over 100 episodes) of those scores is at least **+0.5**.

# Getting Started
Download the environment from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here] https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
- Mac OSX: [click here] https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
- Windows (32-bit): [click here] https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
- Windows (64-bit): [click here] https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip

Place the file in the DRLND GitHub repository, in the p3_collab-compet/ folder, or your project working directory, and unzip (or decompress) the file.

# Dependencies

- Dependencies
- Python 3.6
- Pytorch
- Unity ML-Agents

# Files

- README.md - This file
- Report.md - A report containg the details of the implementation
- Tennis.ipynb - A Jupyter Notebook containing the training code for the two Agents
- model.py - The pytorch neural network code used for training the Actor and the Critic 
- maddpg_agent.py - Definitions for the Agent and its actions
- results_actor01_ckpt.pth - The saved status of the first agent's actor network
- results_critic01_ckpt.pth - The saved status of the first agent's critic network
- A plot of the score over the training episodes is included in the Tennis .ipynb notebook
- requirements.txt - is a list of the model dependcies which can be used to build a new environment in conda
- score-*.PNG* files - Snapshots of the learning environment

# Instructions

You can use the requirements.txt file to create a local environment with all the modules required to tun this project.
Follow the instructions in Tennis.ipynb to get started with training your own agent!

To load the agent into your code python do: 
- from maddpg_agent import Agent

