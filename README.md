# Unity 3D Balance Ball using Deep Deterministic Policy Gradients

[DEMO 12 agents](https://drive.google.com/open?id=1NZsD6lsZkXGo__52LOWXDfMd7p2JazPd) 

[DEMO single agent](https://drive.google.com/open?id=1l1NUW-JZlB4zG2fH9xgqEWcxh_Pwy9pv)
### Motivation 
Reinforcement learning is a method for a program to learn an unknown environment and get some rewards by exploring and gaining success 
or failure experiences. This is similar to the way how we perform in a game, where we explore the game environment and try to achieve 
the goal and get maximum rewards.
There are already successful implementations of reinforcement learning on classical games such as board games, Pacman, sudoku etc. 
Although games with limited states and simple actions (like up/down) can be learned easily by maintaining a state-action value table, it is
not straightforward to see performance on games with large state space and continuous actions. Different approaches have been implemented
on such games to simulate and explore approximately.

###	Game Environment
This project uses an open-source Unity plugin (ML-Agents) for the game environment. [ML-Agents](https://github.com/Unity-Technologies/ml-agents)
provides a simple-to-use environment for training AI agents. Python APIs provided by ML-Agents have been used to get information from the 
game environment.

**3D Balance Ball** is built in Unity 3D environment and contains 12 platform-ball pairs. The agents are the platforms (which are all same
copies of each other). For each platform, it can rotate by x-axis or z-axis to balance its ball as long as possible and keep it from falling
down. In this environment, a platform receives rewards for every step that it balances or drops the ball. The goal of the training process
is to have the platforms learn to balance the ball for a relatively long period of time.

The 12 agents who share a same brain can act independently and can reset themselves randomly when needed. 3D Balance Ball uses the 
continuous state space vector of size 8: the x and z components of the platform's rotation and the x, y, and z components of the ball's
relative position and velocity. The state space is returned from the environment as a stacked vector. The game has a  continuous action
space vector of size 2: the x and z components of action control which can vary continuously. Each agent receives a small positive
reward (+0.1) for each step it keeps the ball on the platform and a larger, negative reward (-1.0) for dropping the ball. An agent is also
assigned  a flag as done when it drops the ball so that it will reset itself with a new ball in a relatively random position for the next 
simulation step.

### Deep Deterministic Policy Gradient (DDPG)
DDPG [T. Lillicrap et al., 2015] is a model-free, off-policy, actor-critic algorithm that uses deep function approximators capable of 
learning policies in problems that have continuous action spaces. It is based on the Deterministic Policy Gradient (DPG) algorithm. 
Where Policy Gradient (PG) doesn’t bother to calculate the ‘value’ to choose the action, rather it chooses actions directly. By adding
neural networks on PG, we can choose actions conveniently through a continuous action range. In algorithms of Q-Learning and Deep Q 
Network (DQN)[Minh et. al., 2013], agents choose action by a value. 

_Why Q-Learning or DQN does not work here_ : 
Q-Learning [Sutton and Barto, 1998]  is an off-policy, model-free RL algorithm. The goal is
to maximize the Q-value through two steps (i) policy iteration and (ii) value iteration. For every possible state,  every possible action
is assigned a value which is a function of the immediate reward for taking that action and the expected reward in future. SARSA is a 
similar learning algorithm that is on-policy hence it learns the Q-value via action performed by the current policy instead of a greedy 
policy. Q-Learning and SARSA do not have the ability to estimate value for unseen states. 

DQN Learning  overcomes this limitation by 
using neural networks to estimate the Q-value function. Although DQN can handle high dimensional observation spaces, it is limited to 
problems where the action space is discrete. Tasks that require continuous control are not suitable for DQN. Discretizing the action 
space in order to apply DQN to the problem results in an extremely large action space. Moreover, the number of actions increases 
exponentially with respect to the freedom degree. Deep Deterministic Policy Gradient (DDPG) overcomes DQN’s limitations by using 
actor-critic architecture
