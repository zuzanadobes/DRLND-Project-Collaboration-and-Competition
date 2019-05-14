### Project criteria

The goal of this coding project is to train two seperate agents to play a game of tennis with each other, competing and cooperating together. The implementation uses a UnityML based agent.  The agents must get a taget score of +.5, as the average, over 1000 consecutive episodes, and over both agents.  After each episode, we add up the rewards that each agent received, to get their average scores, and then take the average of these.   This is used to compute the episode average score.  The environment is considered solved, when the average  score goes over the target score.

When the agent hits the ball over the net, it receives a reward of +0.1. If the ball hits the ground or goes out of bounds, the agent receives a reward of -0.01. Thus the goal of each agent is to keep the ball in play.

Given this information, each agent has to learn how to best select actions. The action space has 2 dimensions; each entry corresponds to movement toward or away from the net, and jumping.

The task is episodic, and in order to solve the environment, the agent must get an average score of +0.5 over 1000 consecutive episodes (after taking the maximum over both agents).

### Code Framework 

The code is written in PyTorch and Python 3.
This repository contains code built on top of the ml-agents from Unity [ Unity Agents ].
The specific environment for this project came in the form of a windows executable file,  is called [Reacher]
(https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

# State Space
- The state space has 8 dimensions (corresponding to the position and velocity of the ball and racket). 

# Vector Observation Space
  - MM variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
  - Visual Observations: None.
  
- Reset Parameters: goal size, and goal movement speed.

### Model Weights

The project includes the saved model weights of the successful agent.
The saved model weights of the successful actor for agent 01 are located in the file: results/results_actor01_ckpt.pth'
The saved model weights of the successful actor for agent 02 are located in the file: results/results_actor02_ckpt.pth'
The saved model weights of the successful critic for critic 01 are located in the file: results/results_critic01_ckpt.pth'
The saved model weights of the successful critic for critic 02 are located in the file: results/results_critic02_ckpt.pth'

###  Hyperparamters 

The following hyperparamter values were released for the Udacity project approval.

The agent starts learning every 5 steps. 
Ornstein-Uhlenbeck noise is used.
#MAXTIMESTEP = 2000     # set in the main program tennis.ipynb - ddpg, the max number of steps in each episode
BUFFER_SIZE = int(1e6)  # replay buffer size - also tried 1e5
BATCH_SIZE = 128        # minibatch size
LR_ACTOR = 1e-3         # learning rate of the actor - also tried 1e-4
LR_CRITIC = 1e-3        # learning rate of the critic - also tried 3e-4
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 1         # learning timestep interval
LEARN_NUM = 5           # number of learning passes
GAMMA = 0.99            # discount factor
TAU = 8e-3              # for soft update of target parameters
EPS_START = 5.0         # initial value for epsilon in noise decay process in Agent.act()
EPS_EP_END = 300        # episode to end the noise decay process
EPS_FINAL = 0           # final value for epsilon after decay
OU_SIGMA = 0.2          # Ornstein-Uhlenbeck noise parameter, volatility
OU_THETA = 0.15         # Ornstein-Uhlenbeck noise parameter, speed of mean reversion

### Policy-based approach

The Reacher environment uses multiple agents, performing the same task, which are all controlled by a single "brain".  The DDPG algorrithm that is used in this project,  collects information from the observation space, using a method called the policy gradient method. For problems with continuous action space as is the case in our problem, a policy oriented approach is used more than value-based approaches. To help make the policy obtaining process more effective a Actor-Critic, where the Critic helps the Actor with performance related information, while the Actor learns.

### Agents

The repository includes functional, well-documented, and organized code for training the agent.

- Agents correspond to a double-jointed arm which can move to target locations.
- Goal: Each agent must move its hand to the goal location, and keep it there.
- Agent Reward Function: A reward of +0.1 is given out for each timestep when the agent's hand is in the goal location. 
- The goal of the agent is to maintain the target location for as many time steps as possible.
- The single agent slution must achieve a score of +13 averaged across all the agents for 100 consecutive episodes.
- The multiple agent solution needs to achieve a moving average score of 30 over all agents, over all episodes.

### Action space 

The action space in this experiment is "continuous" since the agent is executing fine range of movements, or action values, and not just four simple actions.  In the Udacity class there were a number policy-based methods introduced. We try and learn an optimal stochastic policy.   Policy-based methods directly learn the optimal policy, without having to storing and maintaining
all action values and the value function estimatation.   

Vector Action space: Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Learning Algorithm: Deep Deterministic Policy Gradients (DDPG) 

The report clearly describes the learning algorithm, along with the chosen hyperparameters. It also describes the model architectures for any neural networks.

This project uses the DDPG algorithm with a replay buffer. Clearly the DQN algorithm is not sufficient to solve this problem.

The replay buffer corresponds to a memory were the agents can store previous experiences. In our approach the agent randomly selects a fixed number of experiences (i.e., a minibatch of experiences) to update the neural network at a (specific) time step. This is done to  to deal with the problem of temporal correlation, since it allows a mixing of old and new experiences.  But the approach taken here is that all samples are created equal and no experiences are more valuable compared with others. 

The agent is initialized using the hyperparameters are initialized in "ddpg_agent.py".
The parameters which were most influential to the agent's performance included BATCHSIZE and the neural network parameters. 
In general it is very time consuming to measure the effects. 

- Set-up: 2 Agents, using one common Brain. Brains manage vector observation space and action space.
- Agent updates the replay buffer with each experience, which is shared by all agents
- Update the local actor and critic networks using replay buffer "samples"
- the update strategy:  
-- Every T time step ==> X times in a row (per agent) == Using S different samples from the replay buffer.
--- Uses gradient clipping when training the critic network
--- I tried various alternativ update strategies - and will be more precise in the second version of this code as to the differences. In this version it was pure survival coding.
---- Every T time steps update the network 20 times in a row
---- Update the networks 10 times after every 20 timesteps. 

### The neural network 

The network comprises of 2 networks and the settings are described in the model.py file. 
The network architecture used by the Actor and Critic consist of three fully connected layers, with 256 units and 128 units. 
The **Actor** uses the ReLU activation functions & Critic uses the LeakyReLU activation function.
tanh on the output layer.

You can observe the slight difference between the Actor and the Critic networks.

The Actor model is a neural network with3 fully connectedl layers , 2 hidden layers of 512 and 256 units, I also tried 256 units and 128 units respectively. 
- self.fc1 = nn.Linear(state_size*2, fc1_units)
- self.fc2 = nn.Linear(fc1_units, fc2_units)
- self.fc3 = nn.Linear(fc2_units, action_size)
Tanh is used in the final layer that maps states to actions.  

Critic take into account actions of agent1 and agent2. The Critic network is similar to Actor network except the final layer is a fully connected layer that maps states and actions to Q-values. 

- self.fcs1 = nn.Linear(state_size*2, fcs1_units)
- self.fc2 = nn.Linear(fcs1_units+**(action_size*2)**, fc2_units)  <== 
- self.fc3 = nn.Linear(fc2_units, 1)
A dropout of 0.2 is added before the output layer to help the network to learn more efficiently and avoid overfitting.

Noise was added using an Ornstein-Uhlenbeck process theta and sigma were set as the recommended values from classroom reading. (See Also)

Some of the hyper parameters used, several approaches were tried:
* Replay batch size 512
* Buffer size 1e6
* Replay without prioritization
* TAU from  1e-3
* Learning rate 1e-4 for actor and 3e-4 for critic
* Ornstein-Uhlenbeck noise
 
### Reward plot

A plot of rewards per episode is included to illustrate that either:

Results from the "Reacher" environment, 20 agents, goal of average above 30, 100 episodes can be found in the analytics notebook ContinuousControl.ipynb

(Option) The number of episodes needed to solve the environment: ###
(Option) The top mean score achieved while solving the environment: ###
(Option) A plot of rewards per episode is included to show rewards received as the number of episodes reaches: ###

### Ideas for Future Work

Concrete future ideas for improving the agent's performance could include:
(Option) 1. A replay buffer with some kind if prioritization scheme
(Option) 2. Better exploration

## Future Improvements
There are many possible directions to try. Wish there was more time. So before anything I would just try and play with the network layers, and change different aspects of the network, units per layer, or number of layers. Not enough time left to experiment. Other algorithms I would try, which were also covered by the Udacity cource:
- I would try out the [ Priority Experienced Replay algorithm ] 
- I would try the D4G algorithm (refence bellow).  
- Currently the two agents we have in our project use the same network.  We could try an alternative version where we train the 2 agents with seperate separate Actor / Critic networks. 

## See also
You can view the publication from DeepMind here
[ Unity Agents ] https://github.com/Unity-Technologies/ml-agents

[ Priority Experienced Replay algorithm ]  https://ieeexplore.ieee.org/document/8122622

[ Solving Continuous Control environment using Deep Deterministic Policy Gradient (DDPG) agent ] https://medium.com/@kinwo/solving-continuous-control-environment-using-deep-deterministic-policy-gradient-ddpg-agent-5e94f82f366d

[ Learning to play Tennis from scratch with self-play using DDPG ] https://medium.com/@kinwo/learning-to-play-tennis-from-scratch-with-self-play-using-ddpg-ac7389eb980e

[ Andreas Windisch's git repository ] https://github.com/a-windisch/deep_reinforcement_learning_play_tennis ]

[ Henry Chan's git Repository ] https://github.com/kinwo/deeprl-tennis-competition
