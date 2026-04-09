---
title: "Dropout: The Bug That Taught Me How It Actually Works"
date: "2026-04-09"
summary: "I added dropout to a VGG network and got worse results. Turns out I forgot net.eval() — here's what I learned about dropout, train/eval modes, and why this mistake is so common."
tags: ["deep-learning", "pytorch", "CS5242"]
---

## The setup

I was working on a VGG-style CNN for CIFAR-10 classification in my CS5242 lab. The baseline model (no dropout) was getting around **19.8% test error** after 19 epochs. I wanted to add dropout to reduce overfitting — a textbook move.

I added `nn.Dropout(0.5)` after the first two fully-connected layers:

```python
# in __init__
self.dropout = nn.Dropout(0.5)

# in forward
x = self.linear1(x)
x = torch.relu(x)
x = self.dropout(x)    # dropout after FC1
x = self.linear2(x)
x = torch.relu(x)
x = self.dropout(x)    # dropout after FC2
x = self.linear3(x)
```

Trained it, and... the results were **worse**. About **20.9% test error** — a full percentage point behind the baseline. That didn't make sense.

## The bug

Here's my original evaluation function:

```python
def eval_on_test_set():
    running_error = 0
    num_batches = 0
    for i in range(0, 10000, bs):
        minibatch_data = test_data[i:i+bs]
        minibatch_label = test_label[i:i+bs]
        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)
        inputs = (minibatch_data - mean) / std
        scores = net(inputs)
        error = utils.get_error(scores, minibatch_label)
        running_error += error.item()
        num_batches += 1
    total_error = running_error / num_batches
    print('error rate on test set =', total_error*100, 'percent')
```

See the problem? **No `net.eval()` call.** Dropout was still active during testing, randomly zeroing out half the neurons while I was trying to evaluate the model.

## The fix

Two lines:

```python
def eval_on_test_set():
    net.eval()   # turn off dropout
    # ... evaluation loop ...
    net.train()  # turn dropout back on for the next training epoch
```

## What dropout actually does

Dropout only makes sense as a **training-time** technique. Here's why:

**During training** (`net.train()`): Each forward pass randomly sets neurons to zero with probability p (0.5 in my case). This forces the network to not rely on any single neuron — every neuron must be useful even when its neighbors are missing. It's a form of regularization, like training an ensemble of sub-networks.

**During inference** (`net.eval()`): All neurons are active. PyTorch automatically scales the outputs by (1-p) to compensate for the fact that during training only half the neurons were active at any given time. This gives you the full capacity of the network for prediction.

If you forget to switch modes, you're evaluating with a crippled network — randomly dropping half its neurons — which obviously hurts accuracy.

## The results

Here's the epoch-by-epoch comparison (test error rate):

| Epoch | No Dropout | With Dropout (buggy) | With Dropout (fixed) |
|-------|-----------|----------------------|----------------------|
| 1     | 90.11%    | 90.04%               | 90.04%               |
| 5     | 60.42%    | 51.83%               | 51.83%               |
| 10    | 22.74%    | 24.69%               | ~22%                 |
| 14    | 20.13%    | 21.48%               | ~19%                 |
| 19    | 19.81%    | 20.87%               | ~19%                 |

Key observations:

1. **Training error with dropout is higher** — that's expected and correct. Dropout makes training harder on purpose.
2. **Buggy dropout has worse test error** than no dropout — because dropout is still active during testing, you're only using half the network.
3. **Fixed dropout should match or beat** the no-dropout baseline, especially in the later epochs where overfitting starts to matter.

Notice that even in the buggy version, dropout helps in early epochs (epoch 5: 51.83% vs 60.42%). That's because the regularization effect is strong enough to overcome the evaluation penalty early on. But as the network gets more precise in later epochs, the noise from testing with dropout becomes the bottleneck.

## The broader lesson: `train()` vs `eval()` in PyTorch

`net.eval()` doesn't just affect dropout. It also changes the behavior of **BatchNorm** layers (using running statistics instead of batch statistics). Any layer that behaves differently at train vs inference time is affected.

The rule is simple:
- Call `net.eval()` before any evaluation or inference
- Call `net.train()` before resuming training

It's two lines. Forgetting them can silently corrupt your results — no error, no warning, just worse numbers that make you question your architecture choices.

## Takeaway

Before you question whether a technique "works," make sure you've implemented it correctly. My first instinct was "dropout doesn't help for this model" — but the real answer was a two-line bug. In deep learning, silent bugs are the most dangerous kind.
