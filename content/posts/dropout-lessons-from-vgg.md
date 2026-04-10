---
title: "Dropout: The Bug That Taught Me How It Actually Works"
date: "2026-04-09"
summary: "I added dropout (p=0.5) to a VGG network and hit two lessons: first, forgetting net.eval() silently breaks your test accuracy; second, dropout isn't a free lunch — it needs the right hyperparameters and enough training time to pay off."
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
x = self.dropout(x)    # dropout after FC1, p=0.5
x = self.linear2(x)
x = torch.relu(x)
x = self.dropout(x)    # dropout after FC2, p=0.5
x = self.linear3(x)
```

Trained it, and... the results were **worse**. About **20.9% test error** — a full percentage point behind the baseline. That sent me down two rabbit holes.

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

Here's the epoch-by-epoch comparison. Dropout rate = 0.5, applied after FC1 and FC2.

**Test error rate:**

| Epoch | LR     | No Dropout | Dropout (p=0.5) |
|-------|--------|-----------|-----------------|
| 1     | 0.25   | 90.11%    | 90.04%          |
| 2     | 0.25   | 81.14%    | 88.36%          |
| 3     | 0.25   | 76.55%    | 69.55%          |
| 4     | 0.25   | 62.37%    | 60.09%          |
| 5     | 0.25   | 60.42%    | 51.83%          |
| 6     | 0.25   | 46.46%    | 43.41%          |
| 7     | 0.25   | 35.89%    | 36.43%          |
| 8     | 0.25   | 32.73%    | 32.85%          |
| 9     | 0.25   | 27.69%    | 27.17%          |
| 10    | 0.125  | 22.74%    | 24.69%          |
| 11    | 0.125  | 23.37%    | 23.85%          |
| 12    | 0.125  | 23.67%    | 23.81%          |
| 13    | 0.125  | 22.51%    | 23.85%          |
| 14    | 0.0625 | 20.13%    | 21.48%          |
| 15    | 0.0625 | 20.37%    | 21.23%          |
| 16    | 0.0625 | 19.99%    | 20.91%          |
| 17    | 0.0625 | 19.93%    | 20.85%          |
| 18    | 0.03125| 19.80%    | 20.85%          |
| 19    | 0.03125| 19.81%    | 20.87%          |

**Training error rate:**

| Epoch | No Dropout | Dropout (p=0.5) |
|-------|-----------|-----------------|
| 1     | 90.16%    | 89.97%          |
| 5     | 57.25%    | 57.89%          |
| 10    | 16.26%    | 14.84%          |
| 14    | 1.55%     | 0.98%           |
| 17    | 0.01%     | 0.00%           |
| 19    | 0.00%     | 0.00%           |

Key observations:

1. **Dropout helps in the early-to-mid epochs** (epochs 3-6). The test error with dropout is consistently lower — 69.55% vs 76.55% at epoch 3, 51.83% vs 60.42% at epoch 5. The regularization effect is strongest here.
2. **The gap closes and reverses in later epochs.** After epoch 10, the no-dropout model pulls ahead. Final result: 19.81% (no dropout) vs 20.87% (dropout) — about 1% worse.
3. **Training error is similar between the two.** Both reach 0% training error by epoch 17-18, so dropout at p=0.5 isn't preventing the network from fitting the training data — it's just not translating into better generalization for this setup.
4. **Dropout isn't a free lunch.** With only 19 epochs, the dropout model may not have had enough time to converge. Dropout typically needs longer training since it effectively makes the optimization landscape noisier. The model might also benefit from a lower dropout rate (e.g., p=0.2 or p=0.3) or additional techniques like data augmentation to see a clear win.

## The broader lesson: `train()` vs `eval()` in PyTorch

`net.eval()` doesn't just affect dropout. It also changes the behavior of **BatchNorm** layers (using running statistics instead of batch statistics). Any layer that behaves differently at train vs inference time is affected.

The rule is simple:
- Call `net.eval()` before any evaluation or inference
- Call `net.train()` before resuming training

It's two lines. Forgetting them can silently corrupt your results — no error, no warning, just worse numbers that make you question your architecture choices.

## Takeaway

Two lessons from one experiment:

1. **Always check `train()` / `eval()` first.** Before questioning whether a technique works, make sure you've implemented it correctly. Forgetting `net.eval()` is a silent bug — no error, no warning, just worse numbers.
2. **Dropout isn't magic.** Even with the correct implementation, dropout (p=0.5) didn't beat the baseline here. Regularization techniques need the right hyperparameters — a lower dropout rate, more training epochs, or pairing with data augmentation might tell a different story. The point of dropout is to reduce the gap between train and test error, but if the model isn't overfitting much to begin with, dropout might just slow down convergence without a clear payoff.
