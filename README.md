# Binary neural networks

Implementation of some architectures from [Structured Binary Neural Networks for Accurate Image Classification and Semantic Segmentation](https://arxiv.org/abs/1811.10413) in Pytorch

## Models

All architectures are based on ResNet18 now.

There are two groups of models:
- torchvision ResNet compatible. The only difference is BasicBlock that is used inside.
- [deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) compatible. Models with small change in the forward method to be easily integrated with deep-person-reid project.


## Note about binary NN training

When we train binary neural networks we usually use quantized weights and activations for forward and backward passes and full-precision weights for update.
That's why usual backward pass and weights update

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

should be changed with:

```python
optimizer.zero_grad()
loss.backward()
for p in list(model.parameters()):
    if hasattr(p, 'original'):
        p.data.copy_(p.original)
optimizer.step()
for p in list(model.parameters()):
    if hasattr(p, 'original'):
        p.original.copy_(p.data.clamp_(-1, 1))
```


## ONNX compatibility:

Some changes were made into models with fusion gate to make them ONNX-compatible.
Models for training use modules with custom backward function, that can't be converted with ONNX, that's why they are changed with simple sign function for inference.
To create inference model you should pass `freeze=True` flag.

## TODO:

Proper initialization for inference models:
- Mean of weights should be merged into batchnorms
- Weights binarization should be done
