
# The learning algorithm

The learning algorithm I implemented is based on the [Udacity exercise](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn) to implement Deep Q-Learning to solve the [OpenAI Gym's LunarLander environment](https://gym.openai.com/envs/LunarLander-v2/). The implementation is a vanilla Deep Q-Network based on the paper ["Human-level control through deep reinforcement learning"](http://dx.doi.org/10.1038/nature14236)[^1] 

### The implementation details

The implementation at hand is a *Q-Learning algorithm* with a *Deep Neural Network* to approximate the optimal action-value function $Q^*(s,a)$. We are using the L2-Norm for computing the loss and the Adam optimizer[^2]. Additionally we use the two features *Experience Replay* and *Fixed Q-Targets*. The following pseudo-code of the algorithm is a variation of the combination of the paper ["Human-level control through deep reinforcement learning"](http://dx.doi.org/10.1038/nature14236)[^1] and the [Udacity cheatsheet](https://github.com/udacity/deep-reinforcement-learning/blob/master/cheatsheet/cheatsheet.pdf):

Input: policy $\pi$, positive integer *num_episodes*, small positive fraction $\alpha$, GLIE {$\epsilon_i$} with update rule $\epsilon_{i+1} \leftarrow \max(\epsilon_{decay}*\epsilon_i, \epsilon_{min})$
Output: value function $Q$ ($\approx q_\pi$ if *num_episodes* is large enough)

Initialize action-value function $Q(s,a)$ with random weights $\Theta$ for all $s \in S$ and $a \in A(s)$.
Initialize target action-value function $\tilde{Q}(s,a)$ with weights $\tilde{\Theta} = \Theta$ from $Q(s,a)$ for all $s \in S$ and $a \in A(s)$.
Initialize replay memory $D$ to capacity $N$  

**for** $i \leftarrow 1$ to *num_episodes* **do** 
>$\epsilon \leftarrow \epsilon_i$  
Observe $S_0$  
$t←0$  
**repeat**  
>>Choose action $A_t$ using an $\epsilon$-greedy policy derived from $Q$  
Take action $A_t$ and observe $R_{t+1} , S_{t+1}$  
Store experience $(S_t, A_t, R_t, R_{t+1})$ in $D$  
**Every** C steps **do**  
>>>Sample a random minibatch of experience tuples $(S_t, A_t, R_t, R_{t+1})$ from $D$  
Set $Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1} + \gamma \max_a \tilde{Q}(S_{t+1}, a) − Q(S_t, A_t))$  
Perform back propagation according to the L2-Norm  
and the Adam-optimizer to update the weights of the action-value function $Q$  
Update $\tilde{\Theta} \leftarrow \tau*\Theta + (1 - \tau)*\tilde{\Theta}$  

>>**end**  
$t \leftarrow t+1$  

>**until** $S_t$ is terminal;  

**end**  
**return** $Q$  

We use *Experience Replay* to prevent correlations of successive experiences. We can make use of rare occurrences of experiences and learn from them multiple times.  At every time step $t$ we store the experience tuple $(s_j, a_j, r_j, s_{j+1})$ in the replay buffer $D$ and sample from it a random minibatch to update the network. 
We use *Fixed Q-Targets* to prevent correlations which come the fact that we try to update the approximated action-value function $Q$ with an approximated action-value function $Q$. To make use of the feature of fixed Q-targets, we introduce a target action-value function $\tilde{Q}$ and use it in the update rule $Q(S_t, A_t) = Q(S_t, A_t) + \alpha(R_{t+1} + \gamma \max_a \tilde{Q}(S_{t+1}, a) − Q(S_t, A_t))$. Every C steps, we update the parameters of the target action-value function $\tilde{Q}$ by setting $\tilde{\Theta} = \tau*\Theta + (1 - \tau)*\tilde{\Theta}$.

### The hyperparameters

	BUFFER_SIZE = int(1e5)                      # replay buffer size 	
	BATCH_SIZE = 64                             # minibatch size
	GAMMA = 0.99                                # discount factor
	TAU = 1e-3                                  # for soft update of target parameters
	LR = 5e-4                                   # learning rate
	C = 4                                       # how often to update the network
	num_episodes = 2000                         # the number of episodes
	eps_start = 1                               # the initial value of epsilon for the epsilon-greedy policy
	eps_decay = 0.995                           # the rate of change of epsilon for each episode
	eps_min = 0.01                              # the minimum value of epsilon
	

### The model architecture of the deep neural network

We use a 3-layer neural network which takes the 37-dimensional state as the input and outputs the predicted Q-values for each of the 4 actions. All the layers are fully connected. The first layer has 64 nodes and uses a ReLU-activation function. The second layer has 64 nodes and uses a ReLU-activation function as well. The third layer has 4 nodes which correspond to the number of possible actions. We don't use an activation function here as we want to predict the discounted sum of future rewards. Here is the implementation in PyTorch:
```python
class QNetwork(nn.Module):
   """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):

        x = self.fc1(state)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
                
        return x
```
### The performance of the learning algorithm

The algorithm at hand achieves an average reward of 13 over 100 consecutive episodes in 553 episodes. 

![score](https://github.com/fjonck/Project_1_Navigation/blob/master/score.png)


### Ideas for future work

The performance could be improved by using the following methods from the literature: Double DQN[^3], Dueling DQN[^4] or Prioritized Experience Replay[^5].


[^1]: Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., Graves, A., Riedmiller, M., Fidjeland, A. K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S. & Hassabis, D. (2015). Human-level control through deep reinforcement learning. _Nature_, 518, 529--533.

[^2]: Kingma, D. P. & Ba, J. (2014). Adam: A Method for Stochastic Optimization (cite arxiv:1412.6980 Comment: Published as a conference paper at the 3rd International Conference for Learning Representations, San Diego, 2015)

[^3]: van Hasselt, H., Guez, A. & Silver, D. (2015). Deep Reinforcement Learning with Double Q-learning. arXiv:1509.06461v3

[^4]: Wang, Z., de Freitas, N. & Lanctot, M. (2015). Dueling Network Architectures for Deep Reinforcement Learning.. _CoRR_, abs/1511.06581.

[^5]: Schaul, T., Quan, J., Antonoglou, I. & Silver, D. (2015). Prioritized Experience Replay (cite arxiv:1511.05952Comment: Published at ICLR 2016)

> Written with [StackEdit](https://stackedit.io/).
