### Project criteria

The goal of this coding project is to train two seperate agents to play a game of tennis with each other, competing and cooperating together. The implementation uses a UnityML based agent.  The agents must get a taget score of +.5, as the average, over 100 consecutive episodes, and over both agents.  After each episode, we add up the rewards that each agent received, to get their average scores, and then take the average of these.   This is used to compute the episode average score.  The environment is considered solved, when the average  score goes over the target score.

When the agent hits the ball over the net, it receives a reward of +0.1. If the ball hits the ground or goes out of bounds, the agent receives a reward of -0.01. Thus the goal of each agent is to keep the ball in play.

Given this information, each agent has to learn how to best select actions. The action space has 2 dimensions; each entry corresponds to movement toward or away from the net, and jumping.

The task is episodic, and in order to solve the environment, the agent must get an average score of +0.5 over 100 consecutive episodes (after taking the maximum over both agents).

### Code Framework 

- The code is written in PyTorch and Python 3.
- This repository contains code built on top of the ml-agents from Unity [ Unity Agents ].
- The specific environment for this project came in the form of a windows executable file,  is called [Reacher]
(https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)

# State Space / Observation Space
- The state space has 8 dimensions (corresponding to the position and velocity of the ball and racket). 
- Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Model Weights

The project includes the saved model weights of the successful agent.
The saved model weights of the successful actor for agent 01/02 are located in the file: results_actor_ckpt.pth'
The saved model weights of the successful critic for critic 01/02 are located in the file: results_critic_ckpt.pth'

###  Hyperparamters 

The report clearly describes the hyperparamters for the learning algorithm.  The following hyperparamter values were released for the Udacity project approval.

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

### Learning Algorithm: MADDPG (Multi-Agent Deep Deterministic Policy Gradient)

The report clearly describes the learning algorithm. 

This project uses the DDPG algorithm with a replay buffer. This approach is ideas for learning policies in in high-dimensional,continuous action spaces. It is a model-free, actor-critic algorithm using deep neural network function approximators that can learn policies.  

The replay buffer corresponds to a memory were the agents can store previous experiences. In our approach the agent randomly selects a fixed number of experiences (i.e., a minibatch of experiences) to update the neural network at a (specific) time step. This is done to  to deal with the problem of temporal correlation, since it allows a mixing of old and new experiences.  But the approach taken here is that all samples are created equal and no experiences are more valuable compared with others. 

The original DDPG algorithm from which I extended to create the MADDPG version, is outlined in this paper, Continuous Control with Deep Reinforcement Learning, by researchers at Google Deepmind. In this paper, the authors present "a model-free, off-policy actor-critic algorithm using deep function approximators that can learn policies in high-dimensional, continuous action spaces." They highlight that DDPG can be viewed as an extension of Deep Q-learning to continuous tasks.

For the DDPG foundation, I used this vanilla, single-agent DDPG as a template. Then, to make this algorithm suitable for the multiple competitive agents in the Tennis environment, I implemented components discussed in this paper, Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments, by Lowe and Wu, along with other researchers from OpenAI, UC Berkeley, and McGill University. Most notable, I implemented their variation of the actor-critic method (see Figure 1), which I discuss in the following section.

Lastly, I further experimented with components of the DDPG algorithm based on other concepts covered in Udacity's classroom and lessons. My implementation of this algorithm (including various customizations) are discussed below.

The agent is initialized using the hyperparameters are initialized in "ddpg_agent.py".
The parameters which were most influential to the agent's performance included BATCHSIZE and the neural network parameters. 
In general it is very time consuming to measure the effects. 

- Set-up: 2 Agents, using one common Brain. Brains manage vector observation space and action space.
- Agent updates the replay buffer with each experience, which is shared by both agents
- Update the local actor and critic networks using replay buffer "samples"
- the update strategy:  
-- Every T time step ==> X times in a row (per agent) == Using S different samples from the replay buffer.
--- Uses gradient clipping when training the critic network
--- I tried various alternativ update strategies - and will be more precise in the second version of this code as to the differences. In this version it was pure survival coding.
---- Every T time steps update the network M times in a row
---- Update the networks U times after every T timesteps. 

### The neural network 
The report clearly describes the model architectures for the underlying neural networks.

The network comprises of 2 networks and the settings are described in the model.py file. 
The network architecture used by the Actor and Critic consist of three fully connected layers, with 256 units and 128 units. 
The **Actor** uses the ReLU activation functions & Critic uses the LeakyReLU activation function.
tanh on the output layer.

There are slight difference between the Actor and the Critic networks.

The Actor model is a neural network with3 fully connectedl layers , 2 hidden layers of 512 and 256 units, I also tried 256 units and 128 units respectively.  Tanh is used in the final layer that maps states to actions.  

Critic take into account actions of agent1 and agent2. The Critic network is similar to Actor network except the final layer is a fully connected layer that maps states and actions to Q-values. The inner layer accounts for actions possibilities of both agents. 
Noise was added using an Ornstein-Uhlenbeck process theta and sigma were set as the recommended values from classroom reading. 

We use both a local and target networks where one set of parameters is used to select the best action, and another set of parameters (w, vs, w') is used to evaluate that action. In this project, local and target networks are implemented separately for both the actor and the critic.

### Actor-Critic

- The Agent() class can be found in maddpg_agent.py of the source code. 
- The actor-critic models can be found via their respective Actor() and Critic() classes here in models.py.
- Actor-critic methods leverage the strengths both policy-based and value-based methods.
- Each agent  takes actions based on its own observations of the environment.

Using a policy-based approach, the agent (actor) learns how to act by directly estimating the optimal policy and maximizing reward through gradient ascent. Meanwhile, employing a value-based approach, the agent (critic) learns how to estimate the value (i.e., the future cumulative reward) of different state-action pairs. Actor-critic methods combine these two approaches in order to accelerate the learning process. Actor-critic agents can be more stable than value-based agents, while requiring fewer training samples than policy-based agents.

### Reward plot

A plot of rewards per episode is included in the tennis.ipynb notebook to illustrate:

(Option) The number of episodes needed to solve the environment: ###
(Option) The top mean score achieved while solving the environment: ###
(Option) A plot of rewards per episode is included to show rewards received as the number of episodes reaches: ###

## Ideas for Future Improvements
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
