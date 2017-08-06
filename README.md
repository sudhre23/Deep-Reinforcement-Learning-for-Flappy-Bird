# Deep Reinforcement Learning for Flappy Bird

This implementation explores Deep-Q-Learning with the core idea taken from the nature paper titled "Human-level control through deep reinforcement learning"
link : http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html?foxtrotcallback=true

The ideas for implementing Priority Sampling for the flappy bird game was taken from the paper "PRIORITIZED EXPERIENCE REPLAY"
link : https://arxiv.org/pdf/1511.05952.pdf

The ideas for implementing Double-Deep-Q-Learning was taken from the paper "Deep Reinforcement Learning with Double Q-Learning"
link : https://arxiv.org/pdf/1509.06461.pdf

Sincere gratitude to Yan Lau Pau from where the wrapper for the flappy bird pygame was taken and the ideas for the initial base implementation of Deep Reinforcement Learning using Keras. (https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html)

Big thanks to the online blog nervana for assistance in helping to understand Deep Q Learning better. 
link : https://www.intelnervana.com/demystifying-deep-reinforcement-learning/


## Instructions for running the application

1. Clone the complete git directory. The code in Double DQN and Proirity sampling is meant for training the flappy bird. An example command to run the code would be: 

python (filename.py)

1. The DQN.pycan be used in run mode and train mode. Use -Train or -Run to pass as arguements from the command line. 

1. All implementations in training mode give random actions to the flappy bird for the first 3000 iterations in order to populate the experience replay memory, beyond which training starts. 

I am in the process for writing a blog on further details of the implementation. The details of the implementation will be up on this page as soon as the site is up. 
