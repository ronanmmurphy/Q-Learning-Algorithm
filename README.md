# Q-Learning-Algorithm
Reinforcement Learning implementmentation of deterministic FrozenLake ‘grid world’ problem where Q-learning agent learned a defined policy to optimally navigate through the lake. Python was used to program two classes which setup the state and agent respectively. Q-values are set state-action pairs and the algorithm chooses an optimal action for the current state based on estimates of this value. The reward and next state for this action is observed which allows for the Q value to be updated. Over many epochs this algorithm can learn the best path to take for this problem as long as the strategy balances exploration and exploitation correctly.

Grid:

![Grid](https://github.com/ronanmmurphy/Q-Learning-Algorithm/blob/main/Images/grid.PNG?raw=true)

Method:
The Q learning algorithm implements an epsilon greedy method, choosing random value 10% of the time and the best action for all others. The Q value is determined with the formula Q:
Q-Value = (1-α)*Q[(i,j,action)] + α*(Reward + ϒ*Qmax[nxtStateAction])
The values are determined for each step in an episode and updated to the Q table. If the state is an end state the Q value is set to the reward value, -5 for loss +1 for win, and these are updated in Q value also, resetting the State to 0,0. 

Optimal Solution:

![Optimal](https://github.com/ronanmmurphy/Q-Learning-Algorithm/blob/main/Images/optimal_solution.PNG?raw=true)




Rewards Per Episode: 10,000 episodes were run to see the change in reward per epoch over time. It shows that although the algorithm starts off poorly it starts to learn quickly the optimal solution which is takes. As there is still a 10% chance of random action the algorithm never stays on the optimal solution, additional measures could be taken to change the % of exploration over time and exploit the optimal solution.

![Rewards Per Episode](https://github.com/ronanmmurphy/Q-Learning-Algorithm/blob/main/Images/RewardPerEpisode.png?raw=true)
