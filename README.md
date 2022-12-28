<!--
 * @Author: Wenhao Ding
 * @Email: wenhaod@andrew.cmu.edu
 * @Date: 2021-12-21 11:57:44
 * @LastEditTime: 2022-12-28 13:04:26
 * @Description: 
-->
# Generating-by-Discovering (GRADER)

This is the official implementation of NeurIPS 2022 paper [Generalizing Goal-Conditioned Reinforcement Learning with Variational Causal Reasoning](https://arxiv.org/abs/2207.09081).
The released code only contains the Chemistry environment modified from this [repo](https://github.com/dido1998/CausalMBRL).

### Setup code environment

The code is tested with Ubuntu 20.04 and Python 3.8.

```bash
# clone the code
git clone https://github.com/GilgameshD/grader.git
cd grader

# create conda environment
conda create -n grader python=3.8
conda activate grader

# install dependency
pip install -r requirement.txt
```

### Run experiment

Run the following script to train and test agents under different settings.

```bash
# mode - [IID/OOD-S]: environment type
# grader_model - [full/causal/gnn/mlp]: model type
# graph - [collider/chain/full/jungle]: groundtruth graph used in chemistry environment
# exp_name: name of the folder to save results

# one example of training GRADER in IID setting
python train_agent.py --mode IID --grader_model causal --graph chain --exp_name test
```
