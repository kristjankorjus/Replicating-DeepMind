Replicating-DeepMind
====================

Reproducing the results of "Playing Atari with Deep Reinforcement Learning" by DeepMind. All the information is in our [Wiki](https://github.com/kristjankorjus/Replicating-DeepMind/wiki).

Progress: System is up and running on a GPU cluster with cuda-convnet2. It can learn to play better than random but not much better yet :) It is rather fast but still about 2x slower than DeepMind's original system. It does not have RMSprop implemented at the moment which is our next goal. 

Note: Nathan Sprague has a implementation based on Theano. It can do fairly well. See [his github](https://github.com/spragunr/deep_q_rl) for more details.
