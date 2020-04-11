
# Project 1: Navigation

## Introduction

#### The Unity environment
In this project we are solving the "Banana Collector" environment using the open-source Unity plugin **Unity Machine Learning Agents (ML-Agents)**. The main use-case of those environments is to use trained agents for multiple purposes. But we are using it for designing, training and evaluating the performance of our own agent. 

#### The game and the rewards
The goal in the environment "Banana Collector" is to collect as many yellow bananas as possible and avoid blue bananas. This goal is definined by the reward rule +1 for collecting a yellow banana and -1 for collecting a blue banana.

![Banana Collector example](https://video.udacity-data.com/topher/2018/June/5b1ab4b0_banana/banana.gif)

#### The action and state space
The action space consists of four actions: 
- 0: move forward
- 1: move backward
- 2: turn left
- 3: turn right

The state space consists of the agents velocity and ray-based perception of the objects in a specific range of angles in front of the agent. It has 37 dimensions.

#### The instructions for solving the environment
The environment is considered to be solved after having achieved an average score of 13 in 100 consecutive episodes. 

#### The instructions for installing the environment
1. Clone the repository https://github.com/fjonck/Project_1_Navigation
2. Open the notebook Navigation.ipynb with Jupyter Notebook
3. In the menu, click on Cell -> Run All
4. Enjoy watching the Banana Collector Agent learn :)

<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>
