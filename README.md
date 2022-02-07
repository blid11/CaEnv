# CaEnv

The code in this repository can be run at the following Google Colab link: 
https://colab.research.google.com/drive/1FSpolkzsuUehAS59PGGLrWwPhhHQxhx9?usp=sharing

The purpose of the program in this repository is to observe whether or not a reinforcement learning (RL) agent is capable of designing large, stable, propagating patterns in a cellular automaton (CA). The CA in which the agent draws its patterns is a three-dimensional life rule enumerated "5766". This rule is discussed in the paper "The Discovery of a New Glider for the Game of Three-Dimensional Life" written by Carter Bays [1]. In this paper, Bays describes how he discovered the second most common glider in the 5766 CA by running 4 million CA simulations per day [1].

I have personally tried to discover additional gliders in the rule 5766 by running several billion CA simulations per day (my code for which can be found at the link below: https://github.com/blid11/FastRandomSoup.git

Random soups can only sample the massive state space of three-dimensional life rules. The number of ways one can choose 200 cells to set as "alive" in a 10 x 10 x 10 grid is 1000 choose 200 or 6.6 x 10^215 possible ways. Reinforcement learning agents are capable of playing games with large state spaces. The game of Go is estimated to have a state space of 10^172 [2]. If an RL agent can play the game of Go, why should it not be able to play the Game of Life? In the following paragraphs, I will discuss how the state space for three-dimensional life was reduced.

The code in the "Env" class below outlines the game that the agent plays. The agent draws a pattern in the cellular automaton by choosing where to position itself at each time step while a cell is placed every three time steps. The agent can move backward, left, right, up or down. The agent can also choose to stay in its current position. The agent cannot move "forward" due to its boundaries (discussed in a moment). The "Simulation" class will then determine the reward that the agent will receive at each time step.

The agent receives greater reward for making larger, but also more stable patterns. The ideal pattern consists of more than 22 cells, and is capable of moving 50 cells away from the "centre" of the grid where it was drawn. The centre is defined by the coordinate (64,64,64), which is the approximate centre of a 10 x 10 x 10 area where the object is generated. The agent only draws on one half of the 10 x 10 x 10 area, and the cells it draws are reflected to the other half to only produce symmetric patterns. The agent draws in x coordinates 0 to 9, z coordinates 0 to 9 and y coordinates 0 to 4. The pattern is then mirrored so that a cell with a y coordinate of 4 will be drawn at (x, 5, z). The agent starts at position (4,4,4) and so can only move backwards (towards zero) along the y-axis, not forwards. The entire pattern drawn by the agent (and the mirrored cells) is then translated so that the cell (4,4,4) is at (64, 64, 64), and all other cells are shifted by the same vector.

The state space is reduced by allowing only symmetric, semi-contiguous patterns to be drawn by the agent [3]. Semi-contiguous refers to the fact that the agent must place a cell at every third time step, therefore cells must be placed closely together. In a single episode, the agent places 40 cells, and therefore makes 120 actions. The agent has 6 choices for its actions. The state space is therefore reduced to 6^120 or 2.4 x 10^93 possible states.

The simplified Atari games in the reinforcement learning testbed MinAtar use a stack of n 10 x 10 frames as the input or observation to the agents [4]. The largest observation size is 9 x 10 x 10 for the game Seaquest [4]. The game Freeway can have a duration of 2500 frames [4]. All environments in MinAtar use a set of 6 possible actions [4]. With a duration of 120 frames and an observation size of 10 x 10 x 10, the CA environment is similar in complexity to the existing environments in MinAtar [4]. The Deep Q Network and Actor-Critic Network from the MinAtar source code will be the agents that will train in the CA environment [4].

Sources: 

1. Bays C. The discovery of a new glider for the game of three-dimensional Life. Complex Systems. 1990 Dec;4(6):599-602.

2. Schadd FC. Monte-Carlo search techniques in the modern board game Thurn and Taxis. Maastricht University: Maastricht, The Netherlands. 2009 Dec 20. p 26.

3. Bays C. A note about the discovery of many new rules for the game of three-dimensional life. Complex Systems. 2006;16(4):381.

4. Young K, Tian T. Minatar: An atari-inspired testbed for thorough and reproducible reinforcement learning experiments. arXiv preprint arXiv:1903.03176.        2019 Mar 7.
