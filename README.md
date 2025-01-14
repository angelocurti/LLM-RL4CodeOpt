# LLM-RL4CodeOpt
This repository contains all the material related to my master thesis called "LLMs for Code Optimization". During my work I fine-tuned 3 different LLMs: Code T5 plus, WizardCoder and StarCoder. I employed two differente appproaches(SFT, RLFT) to create models capable to optimize chunks of code.
## Dataset
This folder contains the datasets used during the project and a data loader designed for efficient loading of the data. It includes both optimized and non-optimized code examples, with annotations used during the training process. The data loader script is designed to handle large datasets and ensure smooth feeding of data into the fine-tuning pipeline.
## SFT
This folder contains the code related to the supervised fine-tuning (SFT) of the different LLMs. Supervised fine-tuning is done using labeled datasets where the models are trained on specific code optimization tasks with ground truth examples. For each model there is a separate script which takes care of loading the datasets, configuring the model, and fine-tuning the model on the target tasks.
## RLFT
This folder contains the code for RL-based fine-tuning of the different LLMs. The models are fine-tuned using reinforcement learning(PPO), with rewards given based on the optimization performance. This method is more dynamic compared to supervised fine-tuning and adapts the model to give better code optimization results iteratively. The main scripts in this folder implements the RL pipeline, including reward calculation and policy update.
