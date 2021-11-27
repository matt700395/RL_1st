 #!/bin/bash
#PBS -l nodes=1:gpus=1:ppn=1

source activate rl_env

cd ~/PycharmProjects/RL_1st/reinforcement-learning-kr-v2/2-cartpole/1-dqn

python DQN_train.py

conda deactivate