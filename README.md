# Binary neural networks

Implementation of some architectures from [Structured Binary Neural Networks for Accurate Image Classification and Semantic Segmentation](https://arxiv.org/abs/1811.10413) in Pytorch

## Models

All architectures are based on ResNet18 now.

There are two groups of models:
- torchvision ResNet compatible. The only difference is BasicBlock that is used inside.
- [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) compatible. Models with small change in the forward method to be easily integrated with deep-person-reid project.
