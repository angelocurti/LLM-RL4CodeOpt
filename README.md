# LLM-RL4CodeOpt
This repository contains all the material related to my Master's thesis "LLMs for Code Optimization." Throughout my work, I explored the application of LLM to code optimization task. I fine-tuned two different models, CodeT5-Plus and StarCoder, using two distinct approaches: Supervised Fine-Tuning (SFT) and Reinforcement Learning-based Fine-Tuning (RLFT). The goal was to develop models capable of optimizing code snippets.

## Dataset
The dataset/ directory contains all data used during the thesis project. It includes both optimized and non-optimized code samples. A custom-built data loader is provided to efficiently handle large datasets and ensure smooth and scalable loading for fine-tuning tasks.

## SFT (Supervised Fine-Tuning)
The SFT/ directory contains scripts for the Supervised Fine-Tuning of the two LLMs.

## RLFT (Reinforcement Learning-based Fine-Tuning)
The RLFT/ directory contains the implementation for Reinforcement Learning-based Fine-Tuning. This approach fine-tunes the models using the Proximal Policy Optimization (PPO) algorithm, where signal rewards, derived from execution feedback, guide the learning process.


